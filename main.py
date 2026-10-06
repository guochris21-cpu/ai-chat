"""
个人 AI 聊天网站 —— 后端 (FastAPI)

功能：
  1. POST /api/login   用密码换取 session token（有效期 30 天）
  2. POST /api/chat    校验 token，把对话转发给 DeepSeek，并把流式输出原样转发给浏览器
  3. GET  /health      健康检查（Railway / Render 用）
  4. GET  /            返回前端页面 static/index.html

环境变量（只在后端读取，前端永远拿不到）：
  DEEPSEEK_API_KEY  必填  DeepSeek 的 API Key
  APP_PASSWORD      必填  你自己设的登录密码
  SESSION_SECRET    选填  用来签发 token 的随机字符串；不填则由 APP_PASSWORD 派生
  DEEPSEEK_MODEL    选填  默认 deepseek-flash
  SYSTEM_PROMPT     选填  系统提示词
"""

import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ----------------------------------------------------------------------
# 配置：全部从环境变量读取，代码里不写死任何密钥
# ----------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是一个乐于助人的 AI 助手。请使用用户提问时所用的语言回答；"
    "如果用户发来题目或文档的照片，请先准确识别内容，再作答。",
)

# token 有效期：30 天
TOKEN_TTL_SECONDS = 30 * 24 * 3600

# 签名用的密钥。优先用 SESSION_SECRET；没填就由密码派生。
# （改了密码 => 旧 token 全部失效，相当于"踢掉所有已登录设备"）
_SECRET = (os.getenv("SESSION_SECRET") or ("pw:" + APP_PASSWORD)).encode()

# 请求体大小上限（图片是 base64，体积较大），25 MB
MAX_BODY_BYTES = 25 * 1024 * 1024
# 单次最多带多少条历史消息
MAX_MESSAGES = 80

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-chat")
if not DEEPSEEK_API_KEY or not APP_PASSWORD:
    log.warning("DEEPSEEK_API_KEY 或 APP_PASSWORD 未设置，服务无法正常使用！")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Personal AI Chat", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------------
# Token：无状态签名 token，格式 "过期时间戳.HMAC签名"
# 无状态的好处：服务重启 / 重新部署后，已登录的 token 依然有效，不用数据库。
# ----------------------------------------------------------------------
def _sign(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    exp = str(int(time.time()) + TOKEN_TTL_SECONDS)
    return f"{exp}.{_sign(exp)}"


def verify_token(token: str) -> bool:
    try:
        exp, sig = token.split(".", 1)
        # compare_digest 是恒定时间比较，防止时序攻击
        if not hmac.compare_digest(sig, _sign(exp)):
            return False
        return int(exp) > time.time()
    except Exception:
        return False


def require_auth(authorization: str | None):
    """从请求头 `Authorization: Bearer xxx` 里取 token 并校验，失败直接抛 401。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    if not verify_token(authorization[7:].strip()):
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")


# ----------------------------------------------------------------------
# 登录限流：同一个 IP 10 分钟内最多输错 5 次，防止暴力破解密码
# （存在内存里，重启会清空，对个人使用足够了）
# ----------------------------------------------------------------------
_fail_log: dict[str, list[float]] = {}
FAIL_WINDOW = 600
FAIL_LIMIT = 5


def client_ip(request: Request) -> str:
    # 部署在 Railway/Render 后面，真实 IP 在 X-Forwarded-For 的第一个
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def recent_failures(ip: str) -> list[float]:
    now = time.time()
    arr = [t for t in _fail_log.get(ip, []) if now - t < FAIL_WINDOW]
    _fail_log[ip] = arr
    return arr


# ----------------------------------------------------------------------
# 接口 1：登录
# ----------------------------------------------------------------------
@app.post("/api/login")
async def login(request: Request):
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="服务器未设置 APP_PASSWORD")

    ip = client_ip(request)
    if len(recent_failures(ip)) >= FAIL_LIMIT:
        raise HTTPException(status_code=429, detail="尝试次数过多，请 10 分钟后再试")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求格式错误")

    password = str(data.get("password", ""))
    # 密码在后端校验（恒定时间比较）
    if not hmac.compare_digest(password.encode(), APP_PASSWORD.encode()):
        _fail_log.setdefault(ip, []).append(time.time())
        raise HTTPException(status_code=401, detail="密码错误")

    _fail_log.pop(ip, None)  # 登录成功，清空失败记录
    return {"token": make_token(), "expires_in": TOKEN_TTL_SECONDS}


# ----------------------------------------------------------------------
# 校验前端发来的 messages，避免乱七八糟的数据被转发给 DeepSeek
# content 允许两种形式：
#   1) 字符串（纯文字）
#   2) 数组 [{type:"text",text:..}, {type:"image_url",image_url:{url:"data:image/..."}}]
# ----------------------------------------------------------------------
def validate_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]  # 太长就只保留最近的

    clean = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail="消息 role 不合法")
        content = m.get("content")

        if isinstance(content, str):
            clean.append({"role": m["role"], "content": content})
            continue

        if isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    raise HTTPException(status_code=400, detail="content 格式不合法")
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": str(p.get("text", ""))})
                elif p.get("type") == "image_url":
                    img = p.get("image_url") or {}
                    url = str(img.get("url", ""))
                    # 只接受 base64 data URL，不让后端去请求任意外部地址
                    if not url.startswith("data:image/"):
                        raise HTTPException(status_code=400, detail="图片必须是 base64 data URL")
                    parts.append({"type": "image_url", "image_url": img})
                else:
                    raise HTTPException(status_code=400, detail="未知的 content 类型")
            clean.append({"role": m["role"], "content": parts})
            continue

        raise HTTPException(status_code=400, detail="content 格式不合法")

    # 最后一条必须是用户发的
    if clean[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="最后一条消息必须来自用户")
    return clean


# ----------------------------------------------------------------------
# 接口 2：聊天（流式）
# 浏览器 -> 本接口 -> DeepSeek；DeepSeek 的 SSE 流原样转发回浏览器
# ----------------------------------------------------------------------
@app.post("/api/chat")
async def chat(request: Request, authorization: str | None = Header(default=None)):
    require_auth(authorization)  # 先验证 token
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="服务器未设置 DEEPSEEK_API_KEY")

    # 限制请求体大小
    length = request.headers.get("content-length")
    if length and int(length) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="请求太大，请减少图片数量或清空对话")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求格式错误")

    messages = validate_messages(data.get("messages"))

    # 组装 OpenAI 兼容格式的请求
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",  # Key 只在这里出现
        "Content-Type": "application/json",
    }

    client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
    try:
        # 先发请求并等响应头：这样 DeepSeek 报错时，我们能返回正确的 HTTP 状态码
        upstream = await client.send(
            client.build_request("POST", DEEPSEEK_URL, headers=headers, json=payload),
            stream=True,
        )
    except httpx.HTTPError as e:
        await client.aclose()
        log.error("连接 DeepSeek 失败: %s", e)
        raise HTTPException(status_code=502, detail="连接 DeepSeek 失败，请稍后重试")

    if upstream.status_code != 200:
        body = (await upstream.aread()).decode("utf-8", "ignore")[:500]
        await upstream.aclose()
        await client.aclose()
        log.error("DeepSeek 返回 %s: %s", upstream.status_code, body)
        # 尝试取出 DeepSeek 的错误信息给前端看（里面不含 Key）
        try:
            msg = json.loads(body).get("error", {}).get("message") or body
        except Exception:
            msg = body
        raise HTTPException(status_code=502, detail=f"DeepSeek 错误 {upstream.status_code}: {msg}")

    async def relay():
        """把 DeepSeek 的流式数据逐块转发给浏览器。"""
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 告诉 nginx 类代理不要缓冲，保证逐字输出
        },
    )


# ----------------------------------------------------------------------
# 接口 3：健康检查
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------
# 静态文件：根路径返回前端页面
# ----------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

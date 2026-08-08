"""Ad Storyboard Generator — FastAPI Backend v2

Pipeline:
  1. User inputs product info + uploads reference images + optional video URL
  2. Backend searches competitors, optionally analyzes video
  3. Direct API call to local LLM proxy for SAB + storyboard
  4. Generates storyboard frames via Agnes Image 2.0 Flash
  5. Returns complete storyboard
"""

import asyncio
import base64
import io
import json
import re
import uuid
import urllib.parse
from pathlib import Path
from datetime import datetime

import httpx
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

app = FastAPI(title="Ad Storyboard Generator v2")

BASE_DIR = Path(__file__).parent
GALLERY_DIR = BASE_DIR / "gallery"
GENERATED_DIR = BASE_DIR / "generated"
VIDEO_DIR = BASE_DIR / "video_frames"
for d in [GALLERY_DIR / "models", GALLERY_DIR / "products", GALLERY_DIR / "scenes",
          GENERATED_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Async video task store ──
video_tasks: dict[str, dict] = {}

# ── LLM API ──
YT_DLP = str(BASE_DIR / ".venv/bin/yt-dlp")
LLM_BASE = "http://localhost:20128/v1"
LLM_KEY = "sk-990...646a"
LLM_MODEL = "oc/deepseek-v4-flash-free"
VISION_BASE = "https://api-inference.modelscope.cn/v1"
VISION_KEY = "ms-017f687c-6168-4f7f-a0c1-6ea6ddd45f1c"
VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# ── Agnes API ──
AGNES_API = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_MODEL = "agnes-image-2.0-flash"
AGNES_SIZE = "768x1024"


# ── LLM helper ──

async def call_llm(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """Call LLM via local API. Handles SSE streaming -> extracts final text."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4096,
        "reasoning_effort": "none",
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", f"{LLM_BASE}/chat/completions",
                                json=payload, headers=headers) as resp:
                full_text = ""
                reasoning = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            full_text += delta["content"]
                        if delta.get("reasoning_content"):
                            reasoning += delta["reasoning_content"]
                    except json.JSONDecodeError:
                        pass
                return full_text.strip() or reasoning.strip()
    except Exception as e:
        return json.dumps({"error": str(e)})


async def call_llm_vision(
    system_prompt: str,
    user_prompt: str,
    images: list[str],
    timeout: int = 120,
) -> str:
    """Call vision LLM with image inputs via modelscope (streaming only)."""
    content = [{"type": "text", "text": user_prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img}})

    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {VISION_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", f"{VISION_BASE}/chat/completions",
                                json=payload, headers=headers) as resp:
                full_text = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            full_text += delta["content"]
                    except json.JSONDecodeError:
                        pass
                return full_text.strip()
    except Exception as e:
        return json.dumps({"error": str(e)})


def extract_json(text: str) -> dict | None:
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Search ──

async def search_competitors(query: str) -> str:
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
        results = re.findall(r'uddg=(https?[^&"]+)', r.text)
        seen = set()
        urls = []
        for u in results:
            decoded = urllib.parse.unquote(u)
            if decoded not in seen and len(urls) < 3:
                seen.add(decoded)
                urls.append(decoded)
        if not urls:
            return ""
        # Try to fetch pages and extract meaningful text (skip JS/analytics pages)
        async with httpx.AsyncClient(timeout=8) as c:
            for u in urls[:2]:
                try:
                    resp = await c.get(u, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
                    text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    # Skip if mostly JS/tracking code
                    if len(text) < 500 or 'Google Tag Manager' in text or 'gtag' in text:
                        continue
                    return text[:2000]
                except Exception:
                    pass
        return ""
    except Exception:
        return ""


# Video Analysis (via modelscope Qwen vision models)
async def analyze_video(video_url: str) -> str:
    """Download video, extract frames, analyze via vision LLM."""
    video_path = VIDEO_DIR / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    try:
        # Download
        proc = await asyncio.create_subprocess_exec(
            YT_DLP, "-f", "best[height<=480]", "-o", str(video_path),
            video_url, "--no-playlist", "--max-filesize", "100M",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            return ""

        if not video_path.exists():
            return ""

        # Extract 6 key frames at intervals
        duration = 0
        probe = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await probe.communicate()
        try:
            duration = float(stdout.decode().strip())
        except (ValueError, TypeError):
            duration = 30  # assume 30s

        num_frames = 6
        frame_paths = []
        for i in range(num_frames):
            ts = duration * (i + 0.5) / num_frames
            out = VIDEO_DIR / f"frame_{video_path.stem}_{i}.jpg"
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-ss", str(ts), "-i", str(video_path),
                "-vframes", "1", "-q:v", "2", str(out),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if out.exists():
                frame_paths.append(str(out))

        if not frame_paths:
            return ""

        # Convert frames to base64
        frames_b64 = []
        for fp in frame_paths:
            with open(fp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                frames_b64.append(f"data:image/jpeg;base64,{b64}")

        # Analyze via vision LLM
        analysis = await call_llm_vision(
            "你是一位视频广告分析师。分析这些竞品广告画面的视觉风格、运镜、灯光、色彩、场景构图和情绪基调。",
            f"分析来自竞品广告的 {len(frames_b64)} 帧画面。描述：\n1. 视觉风格和色彩搭配\n2. 布光方式\n3. 镜头角度和景别\n4. 场景构图和道具\n5. 情绪基调\n6. 这条广告为什么有效",
            frames_b64[:2],  # limit to 2 frames for speed
            timeout=120,
        )
        return analysis
    except Exception:
        return ""
    finally:
        # Cleanup video file
        if video_path.exists():
            video_path.unlink()


# ── Agnes ──

async def call_agnes(prompt: str, api_key: str, image_urls: list[str] | None = None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": AGNES_MODEL,
        "prompt": f"{prompt}, 电影级画质, 专业布光, 细节丰富",
        "size": AGNES_SIZE,
        "extra_body": {"response_format": "url"},
    }
    if image_urls:
        payload["extra_body"]["image"] = image_urls
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(AGNES_API, headers=headers, json=payload)
        if resp.status_code != 200:
            return {"error": f"Agnes {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        img_url = data.get("data", [{}])[0].get("url")
        if not img_url:
            return {"error": "No image URL in Agnes response"}
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt[:30]) + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        local_path = GENERATED_DIR / safe_name
        async with httpx.AsyncClient(timeout=60) as dl:
            img_resp = await dl.get(img_url)
            local_path.write_bytes(img_resp.content)
        return {"url": img_url, "local_path": str(local_path)}
    except Exception as e:
        return {"error": str(e)}


# ── Reference Image Helpers ──

def _resize_for_vision(path: str, max_size: int = 2048) -> str:
    """Resize image to fit within max_size×max_size, return data URL."""
    from PIL import Image
    try:
        img = Image.open(path)
        w, h = img.size
        if w <= max_size and h <= max_size:
            mime = "image/png" if path.endswith(".png") else "image/jpeg"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"
        scale = max_size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "PNG" if path.endswith(".png") else "JPEG"
        img.save(buf, format=fmt)
        mime = "image/png" if path.endswith(".png") else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return ""


async def analyze_reference_images() -> dict:
    """Analyze product & scene reference images via vision LLM. Returns descriptions."""
    gallery = get_gallery_images()
    result = {"products": "", "scenes": []}
    for cat in ("products", "scenes"):
        imgs = [i for i in gallery if i["category"] == cat]
        if not imgs:
            continue
        data_urls = []
        for img in imgs[:3]:
            du = _resize_for_vision(str(BASE_DIR / img["path"].lstrip("/")))
            if du:
                data_urls.append(du)
        if not data_urls:
            continue
        if cat == "products":
            prompt = "分析这些产品参考图。描述产品的准确外观：类型、形状、材质、颜色、设计细节、标识。分镜必须严格忠于这些外观，不得添加参考图中不存在的元素。"
        else:
            prompt = "分析这些场景参考图。分别描述每张场景的环境、地点、光线、氛围、色调、标志性元素。分镜中的场景必须严格来自这些参考图，不得虚构参考图中没有的地点或元素。"
        resp = await call_llm_vision("你是产品/场景视觉分析专家，输出中文描述。", prompt, data_urls, timeout=120)
        if cat == "products":
            result["products"] = resp
        else:
            parts = [s.strip() for s in resp.replace("\n\n", "\n").split("\n") if s.strip()]
            result["scenes"] = parts if parts else [resp]
    return result


def build_frame_ref_images(ref_type: str, frame_idx: int, ref_images: dict, scene_desc: str = "") -> list[str]:
    """Build reference image list for one frame. Max 1 scene/model image (rotated), plus product anchor.
    If scene_desc mentions a person and model images exist, the model image is mandatory."""
    images_to_use = []
    cat_map = {"scenes": "scenes", "models": "models", "products": "products"}

    # Force model reference if the description mentions a person and models are uploaded
    if "models" in ref_images and ref_images["models"] and _mentions_person(scene_desc):
        ref_type = "models"

    # Scene/model: at most ONE image. If multiple scene images, rotate by frame index.
    if ref_type in cat_map and ref_type in ref_images:
        refs = ref_images[ref_type]
        if refs:
            idx = frame_idx % len(refs)
            du = image_to_data_url(str(BASE_DIR / refs[idx].lstrip("/")))
            if du:
                images_to_use.append(du)

    # Product anchor: always add first product image for consistency
    if "products" in ref_images and ref_images["products"]:
        du = image_to_data_url(str(BASE_DIR / ref_images["products"][0].lstrip("/")))
        if du and du not in images_to_use:
            images_to_use.append(du)

    return images_to_use


_PERSON_KEYWORDS = ("人", "模特", "主角", "角色", "演员", "背影", "手", "脚", "脸",
                    "顾客", "女孩", "男孩", "女人", "男人", "女子", "男子", "人物",
                    "穿着", "身影", "剪影", "顾客", "用户", "她", "他", "她们", "他们")


def _mentions_person(desc: str) -> bool:
    if not desc:
        return False
    return any(k in desc for k in _PERSON_KEYWORDS)

def get_gallery_images(category: str | None = None) -> list[dict]:
    results = []
    dirs = [GALLERY_DIR / "models", GALLERY_DIR / "products", GALLERY_DIR / "scenes"]
    for d in dirs:
        if category and d.name != category:
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                results.append({
                    "path": f"/gallery/{d.name}/{f.name}",
                    "category": d.name,
                    "filename": f.name,
                })
    return results


def image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((BASE_DIR / "storyboard.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_image(category: str = Form(...), file: UploadFile = File(...)):
    allowed = {"models", "products", "scenes"}
    if category not in allowed:
        return JSONResponse({"error": f"category must be one of {allowed}"}, status_code=400)
    dest = GALLERY_DIR / category / file.filename
    dest.write_bytes(await file.read())
    return {"path": f"/gallery/{category}/{file.filename}", "category": category}


@app.get("/api/gallery")
async def list_gallery(category: str | None = None):
    return {"images": get_gallery_images(category)}


@app.delete("/api/gallery/{category}/{filename}")
async def delete_gallery_image(category: str, filename: str):
    p = GALLERY_DIR / category / filename
    if p.exists():
        p.unlink()
    return {"status": "deleted"}


# ── Generate ──

SYSTEM_SAB = """你是专业的 TVC 广告策略专家。你分析产品并生成 SAB（S=差异化核心, A=信任背书, B=价值共鸣）分级。始终输出结构化 JSON。

TVC 叙事模型（选择最适合的一个）：
A. 痛点-解决：痛点场景 → 产品拯救
B. 产品电影化拆解：多 Phase 微电影，逐步揭示产品卖点
C. 品牌世界穿梭：使用场景 ↔ 产品特写交叉剪辑，匹配剪辑衔接
D. 生活方式短片：产品始终在场景中，通过运镜手法自然突出
E. 情感锚点：情感故事，产品为载体
F. 蒙太奇揭示：视觉奇观 → 产品揭示
G. 前后对比：使用前后的强烈反差
H. 品牌宣言：价值观驱动，产品收束

视觉风格体系：
A. 真人实拍/摄影级：如苹果广告——精确布光、浅景深、自然纹理
B. 真人电影剧照：介于真人和 CG 之间——如漫威电影、权游——戏剧化布光
C. 3A 游戏 CG：高品质游戏 CG 渲染——如最终幻想、原神——风格化但精致
D. 高精 CG 引擎级：虚幻引擎 5 Demo——超写实，近乎照片级
E. 特定美学风格：水墨、赛博朋克、动漫等——明确指定风格

核心规则：
- 无 CTA，无硬推销，无"立即购买"
- 品牌世界 = 产品所在的环境（如运动相机→跳伞/滑雪，豪车→山路/沙漠）
- 一条视频只打一个 S 点，不要堆砌
- S 必须在前 3-5 秒出现

输出格式（必须严格遵守，仅返回以下 JSON，不要有任何其他文字）：
{
  "narrative_model": "选中的叙事模型名称（如品牌世界穿梭）",
  "narrative_model_desc": "一句话解释叙事模型的选择",
  "competitor_analysis": "竞品分析",
  "sab_grading": {"s": "差异化核心", "a": "信任背书", "b": "价值共鸣"},
  "overall_style_guide": "包含品牌世界描述的风格指南"
}
"""

SYSTEM_STORYBOARD = """你是专业的 TVC 分镜导演。你创作电影级广告分镜，包含精确的镜头语言、运镜方向和情绪基调。始终输出结构化 JSON。

镜头语言体系：
[景别·视角] — 每个分镜以"景别 + 视角"开头

景别：
- 大广角全景（建立镜头）
- 全景
- 中景（腰部以上）
- 近景（胸部以上）
- 特写（细节）
- 微距（纹理级别）

视角：
- 平视
- 俯拍（高角度/俯视）
- 仰拍（低角度）
- 侧拍（侧面/侧脸）
- 斜侧（荷兰角/倾斜）
- 过肩（越过肩膀）

运镜方式：
- 固定
- 缓推（缓慢推近）
- 缓拉（缓慢拉远）
- 横移（横向平移）
- 摇摄（上下摇动）
- 跟拍（跟随/跟踪）
- 环绕（弧线/环绕）
- 升降（升降机）
- 手持（手持摇晃）

灯光：
- 侧光
- 逆光（背光/轮廓光）
- 顺光（正面光/平光）
- 顶光
- 底光
- 丁达尔光（体积光柱）
- 柔光
- 硬光
- 低调（低调布光）
- 高调（高调布光）

核心规则：
- 无 CTA，无硬推销
- 品牌世界 = 产品的自然使用环境；产品世界 = 影棚 / 产品细节
- 分镜间景别要有变化，不要重复相同景别
- 每个分镜约 5 秒
- 30 秒广告 = 6 帧，60 秒广告 = 12 帧
- 如果提供了 brand_world，将其融入场景描述
- CRITICAL: scene_description 是必填字段，必须包含完整的画面描述（构图、光影、色彩、质感、静置物体、人物定格姿态、表情）。描述冻结的瞬间——仿佛按下暂停键看到的那一帧。禁止动态动词。scene_description 不能为空。
- motion_notes（可选补充字段）: 如果这帧有动态动作（人物走向、拿起、转身、跳跃等）或运镜节奏，在此描述，供后续视频使用。没有动态动作则留空。
- 始终输出 JSON：{"storyboard": [{"scene":1,"duration_sec":5,"shot_type":"","camera_angle":"","camera_movement":"","lighting":"","scene_description":"","motion_notes":"","dialogue_or_voiceover":"","mood":"","color_palette":"","transition":"cut","reference_type":"products|models|scenes|none"}]}
"""


# ── Async video analysis ──

@app.post("/api/analyze_video")
async def start_video_analysis(video_url: str = Form(...)):
    task_id = uuid.uuid4().hex[:12]
    video_tasks[task_id] = {"status": "pending", "progress": 0, "result": ""}
    asyncio.create_task(_run_video_analysis(task_id, video_url))
    return {"task_id": task_id}


@app.get("/api/analyze_video/{task_id}")
async def get_video_analysis(task_id: str):
    task = video_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return task


async def _run_video_analysis(task_id: str, video_url: str):
    task = video_tasks[task_id]
    task["status"] = "running"
    task["progress"] = 10
    result = await analyze_video(video_url)
    task["result"] = result
    task["status"] = "done" if result else "error"
    task["progress"] = 100


# ── Brief store (cross-step data) ──
briefs: dict[str, dict] = {}


# ── Analyze: creative brief only ──

@app.post("/api/analyze")
async def analyze_storyboard(
    product_name: str = Form(...),
    product_description: str = Form(...),
    audience: str = Form(""),
    duration: int = Form(30),
    narrative_model: str = Form("auto"),
    visual_style: str = Form("photo"),
    brand_world: str = Form(""),
    video_analysis: str = Form(""),
):
    # Step 1: Search competitors
    search_query = f"{product_name} {product_description[:30]} 广告 宣传片"
    competitor_data = await search_competitors(search_query)

    # Step 2: Analyze reference images (products & scenes) if present
    ref_analysis = await analyze_reference_images()
    ref_ctx = ""
    if ref_analysis["products"]:
        ref_ctx += f"\n产品参考分析：{ref_analysis['products'][:500]}\n分镜中产品外观必须严格忠于以上描述，不得添加参考图中不存在的特征。"
    if ref_analysis["scenes"]:
        scene_list = "；".join(ref_analysis["scenes"])[:500]
        ref_ctx += f"\n场景参考分析：{scene_list}\n分镜中的场景必须严格来自这些参考图，不得虚构参考图中没有的地点或元素。"
    ref_ctx = ref_ctx.strip()

    # Step 3: SAB grading via direct LLM API
    sab_prompt = f"""产品：{product_name}
描述：{product_description[:300]}
目标受众：{audience}
叙事模型：{narrative_model}
视觉风格：{visual_style}
品牌世界：{brand_world or '(未指定——请创建一个)'}

{ref_ctx}

竞品信息：{competitor_data[:800]}
视频分析：{video_analysis[:800]}

为品牌 TVC 广告创建 SAB 分级（无 CTA，无硬推销）。
SAB 分级必须严格基于上方"产品参考分析"中描述的产品外观、材质、包装、设计细节，从视觉特征中提炼卖点，不得凭空捏造产品不具备的特征。
从列表中选出最合适的叙事模型。如果为"auto"，自动选择最合适的。
始终输出 JSON，仅包含以下键：
- "narrative_model": 字符串（选中的模型名称）
- "narrative_model_desc": 字符串（一句话解释叙事模型的选择）
- "competitor_analysis": 字符串（竞品分析）
- "sab_grading": {{"s": 字符串, "a": 字符串, "b": 字符串}}
- "overall_style_guide": 字符串（包含品牌世界描述）"""
    sab_response = await call_llm(SYSTEM_SAB, sab_prompt, timeout=120)
    sab_result = extract_json(sab_response) or {"competitor_analysis": "", "sab_grading": {}, "overall_style_guide": ""}

    brief_id = uuid.uuid4().hex[:12]
    briefs[brief_id] = {
        "product_name": product_name,
        "product_description": product_description,
        "audience": audience,
        "duration": duration,
        "narrative_model": sab_result.get("narrative_model", narrative_model),
        "narrative_model_desc": sab_result.get("narrative_model_desc", ""),
        "visual_style": visual_style,
        "brand_world": brand_world,
        "sab_grading": sab_result.get("sab_grading", {}),
        "overall_style_guide": sab_result.get("overall_style_guide", ""),
        "competitor_analysis": sab_result.get("competitor_analysis", ""),
        "video_analysis": video_analysis[:500] if video_analysis else "",
        "ref_ctx": ref_ctx,
        "ref_analysis": ref_analysis,
    }

    return {
        "brief_id": brief_id,
        "narrative_model": sab_result.get("narrative_model", narrative_model),
        "narrative_model_desc": sab_result.get("narrative_model_desc", ""),
        "competitor_analysis": sab_result.get("competitor_analysis", ""),
        "sab_grading": sab_result.get("sab_grading", {}),
        "overall_style_guide": sab_result.get("overall_style_guide", ""),
        "video_analysis": video_analysis[:500] if video_analysis else "",
        "ref_analysis": ref_analysis,
    }


# ── Generate storyboard prompts (step 2: user confirms brief, then generates prompts) ──

@app.post("/api/generate")
async def generate_storyboard(
    brief_id: str = Form(...),
    duration: int = Form(30),
    visual_style: str = Form("photo"),
    brand_world: str = Form(""),
):
    brief = briefs.get(brief_id)
    if not brief:
        return {"error": "brief not found or expired"}

    product_name = brief["product_name"]
    product_description = brief["product_description"]
    audience = brief["audience"]
    narrative_model = brief["narrative_model"]
    narrative_model_desc = brief["narrative_model_desc"]
    sab_grading = brief["sab_grading"]
    overall_style_guide = brief["overall_style_guide"]
    ref_ctx = brief.get("ref_ctx", "")

    sec_per_frame = 5
    num_frames = max(6, min(14, duration // sec_per_frame))
    sb_prompt = f"""产品：{product_name}
描述：{product_description[:200]}
受众：{audience}
时长：{duration}s，{num_frames} 帧
叙事模型：{narrative_model}
叙事说明：{narrative_model_desc}
品牌世界：{brand_world or overall_style_guide[:200]}
视觉风格：{visual_style}
S（差异化）：{sab_grading.get('s','')}
A（信任）：{sab_grading.get('a','')}
B（价值）：{sab_grading.get('b','')}
风格指南：{overall_style_guide}

{ref_ctx}

创作一段 {duration} 秒的 TVC 广告分镜，共 {num_frames} 帧，遵循叙事模型。每帧约 {duration//num_frames} 秒。
- 将品牌世界融入场景描述，用生动的镜头语言呈现
- 不同分镜间景别、视角、运镜要有变化
- 每帧指定灯光
- CRITICAL: scene_description 是必填字段，必须包含完整的画面描述（冻结的瞬间、构图、光影、色彩、质感）。禁止动态动词。scene_description 不能为空。
- motion_notes（可选补充字段）: 把所有动态动作（人物走向、拿起、转身、跳跃等）放在这里，供后续视频使用。没有则留空。
- 无 CTA，无硬推销
始终输出 JSON：{{"storyboard":[{{"scene":1,"duration_sec":5,"shot_type":"","camera_angle":"","camera_movement":"","lighting":"","scene_description":"","motion_notes":"","dialogue_or_voiceover":"","mood":"","color_palette":"","transition":"cut","reference_type":"products|models|scenes|none"}}]}}"""
    sb_response = await call_llm(SYSTEM_STORYBOARD, sb_prompt, timeout=180)
    sb_result = extract_json(sb_response)
    storyboard = sb_result.get("storyboard", []) if sb_result else []
    if not storyboard:
        return {"error": "Storyboard generation failed", "raw": sb_response}

    return {
        "brief_id": brief_id,
        "narrative_model": narrative_model,
        "narrative_model_desc": narrative_model_desc,
        "competitor_analysis": brief.get("competitor_analysis", ""),
        "sab_grading": sab_grading,
        "overall_style_guide": overall_style_guide,
        "video_analysis": brief.get("video_analysis", ""),
        "ref_analysis": brief.get("ref_analysis", {"products": "", "scenes": []}),
        "frames": storyboard,
    }


# ── Generate images for all frames ──

@app.post("/api/generate_images")
async def generate_images(data: dict):
    """Generate Agnes images for all storyboard frames. Accepts frames + api_key + visual_style."""
    frames = data.get("frames", [])
    api_key = data.get("api_key", "")
    visual_style = data.get("visual_style", "photo")
    if not api_key:
        return {"error": "api_key required"}
    if not frames:
        return {"error": "frames required"}

    gallery = get_gallery_images()
    ref_images = {}
    for img in gallery:
        ref_images.setdefault(img["category"], []).append(img["path"])

    result = []
    for frame in frames:
        desc = frame.get("scene_description", "")
        shot = frame.get("shot_type", "")
        angle = frame.get("camera_angle", "")
        lt = frame.get("lighting", "")
        prompt_text = f"{desc}, {visual_style}风格"
        if lt:
            prompt_text += f", {lt}"
        if shot or angle:
            prompt_text += f", {shot}{angle}构图"

        images_to_use = build_frame_ref_images(frame.get("reference_type", "none"), frame.get("scene", 1) - 1, ref_images, frame.get("scene_description", ""))

        agnes_result = await call_agnes(prompt_text, api_key, images_to_use or None)
        result.append({**frame, "image_url": agnes_result.get("url", ""), "error": agnes_result.get("error")})

    return {"frames": result}


# ── Regenerate single frame ──

@app.post("/api/regenerate")
async def regenerate_frame(data: dict):
    """Regenerate a single frame's image. Accepts frame data + visual_style + api_key."""
    desc = data.get("scene_description", "")
    shot = data.get("shot_type", "")
    angle = data.get("camera_angle", "")
    lt = data.get("lighting", "")
    visual_style = data.get("visual_style", "photo")
    api_key = data.get("api_key", "")
    ref_type = data.get("reference_type", "none")

    if not api_key:
        return {"error": "api_key required"}
    if not desc:
        return {"error": "scene_description required"}

    prompt_text = f"{desc}, {visual_style}风格"
    if lt:
        prompt_text += f", {lt}"
    if shot or angle:
        prompt_text += f", {shot}{angle}构图"

    gallery = get_gallery_images()
    ref_images = {}
    for img in gallery:
        ref_images.setdefault(img["category"], []).append(img["path"])

    images_to_use = build_frame_ref_images(ref_type, 0, ref_images, desc)

    result = await call_agnes(prompt_text, api_key, images_to_use or None)
    return {"image_url": result.get("url", ""), "error": result.get("error")}


# ── Static files ──

@app.get("/gallery/{category}/{filename}")
async def serve_gallery(category: str, filename: str):
    p = GALLERY_DIR / category / filename
    return FileResponse(p) if p.exists() else JSONResponse({"error": "not found"}, status_code=404)


@app.get("/generated/{filename}")
async def serve_generated(filename: str):
    p = GENERATED_DIR / filename
    return FileResponse(p) if p.exists() else JSONResponse({"error": "not found"}, status_code=404)


@app.get("/video_frames/{filename}")
async def serve_video_frame(filename: str):
    p = VIDEO_DIR / filename
    return FileResponse(p) if p.exists() else JSONResponse({"error": "not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8021, reload=True)

import { useCallback, useEffect, useRef, useState } from "react";
import { ScanText, Upload, Loader2, ImagePlus, Save, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import { Events } from "@wailsio/runtime";
import { OcrService, RuntimeService } from "@bindings/cpm_orc";

let currentBoxes: any[] = [];

export default function OcrPage() {
  const [tier, setTier] = useState("small");
  const [status, setStatus] = useState("未加载");
  const [paths, setPaths] = useState({ det: "", rec: "", cls: "", ori: "", dict: "" });
  const [showPaths, setShowPaths] = useState(false);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [lines, setLines] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const refresh = useCallback(async () => {
    try {
      const st = await OcrService.Status();
      setStatus(st.loaded ? "已加载" : "未加载");
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const loadPaths = async (t: string) => {
    try {
      const [det, rec, cls, ori, dict] = await OcrService.DefaultPaths("ch", t);
      setPaths({ det, rec, cls, ori, dict });
    } catch (e) { toast(String(e), true); }
  };

  const install = async () => {
    try {
      await OcrService.InstallDefaults("ch", tier);
      await loadPaths(tier);
      toast(`PP-OCRv6 ${tier} 模型已就绪`);
    } catch (e) { toast(String(e), true); }
  };

  const loadOcr = async () => {
    try {
      await OcrService.Load(paths.det, paths.rec, paths.cls, paths.ori, paths.dict);
      await refresh();
      toast("OCR 模型加载成功");
    } catch (e) { toast(String(e), true); }
  };

  const loadDefault = async () => {
    try { await loadPaths(tier); await loadOcr(); } catch (e) { toast(String(e), true); }
  };

  // --- image preview + overlay ---
  const drawBoxes = useCallback((boxes: any[]) => {
    currentBoxes = boxes || [];
    const img = imgRef.current;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!img || !canvas || !wrap || !img.src || !img.naturalWidth) return;
    const wrapRect = wrap.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    canvas.width = Math.round(imgRect.width);
    canvas.height = Math.round(imgRect.height);
    canvas.style.left = `${Math.round(imgRect.left - wrapRect.left)}px`;
    canvas.style.top = `${Math.round(imgRect.top - wrapRect.top)}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const sx = canvas.width / img.naturalWidth;
    const sy = canvas.height / img.naturalHeight;
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2;
    for (const line of currentBoxes) {
      ctx.beginPath();
      for (let i = 0; i < 4; i++) {
        const x = line.box[i][0] * sx;
        const y = line.box[i][1] * sy;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }, []);

  const onImage = useCallback((b64: string, rotation: number, finish: () => void) => {
    const dataUrl = `data:image/png;base64,${b64}`;
    setImageB64(b64);
    setImgUrl(dataUrl);
    const im = new Image();
    im.onload = () => {
      const draw = () => requestAnimationFrame(() => {
        drawBoxes(currentBoxes);
        finish();
      });
      // rotation handled by re-rendering rotated src
      if (rotation) {
        const c = document.createElement("canvas");
        const w = im.naturalWidth, h = im.naturalHeight;
        c.width = rotation === 90 || rotation === 270 ? h : w;
        c.height = rotation === 90 || rotation === 270 ? w : h;
        const ctx = c.getContext("2d")!;
        ctx.translate(c.width / 2, c.height / 2);
        ctx.rotate((rotation * Math.PI) / 180);
        ctx.drawImage(im, -w / 2, -h / 2);
        setImgUrl(c.toDataURL("image/png"));
        setTimeout(draw, 50);
      } else {
        draw();
      }
    };
    im.src = dataUrl;
  }, [drawBoxes]);

  // background OCR events
  useEffect(() => {
    const offResult = Events.On("ocr:result", (e: any) => {
      const d = e.data;
      setLines(d.lines || []);
      if (d.image) onImage(d.image, d.rotation || 0, () => {});
    });
    const offError = Events.On("ocr:error", (e: any) => toast("OCR 失败: " + (e.data?.error || ""), true));
    return () => { offResult(); offError(); };
  }, [onImage]);

  const pickFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => onImage(reader.result!.toString().split(",")[1], 0, () => {});
    reader.readAsDataURL(file);
  };

  const run = async () => {
    if (!imageB64) { toast("请先选择图片", true); return; }
    setBusy(true);
    setLines([]);
    try {
      const res = await OcrService.RecogniseBase64(imageB64);
      setLines(res!.lines || []);
      onImage(imageB64, res!.rotation || 0, () => {});
    } catch (e) {
      toast("识别失败: " + e, true);
    } finally { setBusy(false); }
  };

  const savePng = () => {
    const img = imgRef.current;
    if (!img || !img.src) return;
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = Math.max(2, Math.round(c.width / 400));
    ctx.strokeStyle = "#34d399";
    for (const line of currentBoxes) {
      ctx.beginPath();
      for (let i = 0; i < 4; i++) {
        const x = line.box[i][0], y = line.box[i][1];
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }
    const a = document.createElement("a");
    a.href = c.toDataURL("image/png");
    a.download = "ocr-reconstructed.png";
    a.click();
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">PaddleOCR</h1>
          <p className="mt-1 text-sm text-muted-foreground">PP-OCRv6 多档模型，识别前自动矫正方向，生成重建图片。</p>
        </div>
        <Badge variant={status === "已加载" ? "success" : "secondary"}>{status}</Badge>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="flex items-center gap-2"><Settings2 className="h-4 w-4" />模型配置</CardTitle>
          <div className="flex items-center gap-2">
            <Select value={tier} onValueChange={(v) => { setTier(v); loadPaths(v); }}>
              <SelectTrigger className="w-44"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="tiny">精简 tiny（最快）</SelectItem>
                <SelectItem value="small">均衡 small（推荐）</SelectItem>
                <SelectItem value="medium">精准 medium（最准）</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={install}>安装该档模型</Button>
            <Button variant="outline" onClick={loadDefault}>加载默认</Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-2 flex items-center justify-between">
            <Button size="sm" variant="ghost" onClick={() => setShowPaths(!showPaths)}>
              <Settings2 className="h-4 w-4" />模型路径{showPaths ? "（收起）" : ""}
            </Button>
            <div className="flex gap-2">
              <Button size="sm" onClick={loadOcr}>加载模型</Button>
              <Button size="sm" variant="outline" onClick={() => RuntimeService.OpenFolder(paths.det.split("/").slice(0, -1).join("/"))}>打开目录</Button>
            </div>
          </div>
          {showPaths && (
            <div className="grid grid-cols-2 gap-3">
              {(["det", "rec", "cls", "ori", "dict"] as const).map((k) => (
                <div key={k} className="space-y-1">
                  <Label>{k === "det" ? "检测模型" : k === "rec" ? "识别模型" : k === "cls" ? "文本行方向(可选)" : k === "ori" ? "文档方向(可选)" : "字典"}</Label>
                  <Input value={paths[k]} onChange={(e) => setPaths({ ...paths, [k]: e.target.value })} />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><ImagePlus className="h-4 w-4" />输入图片</CardTitle>
            <div className="flex gap-2">
              <label className="cursor-pointer">
                <Button asChild variant="outline"><span><Upload className="h-4 w-4" />选择图片</span></Button>
                <input type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && pickFile(e.target.files[0])} />
              </label>
              <Button onClick={run} disabled={busy}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanText className="h-4 w-4" />}识别
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div
              ref={wrapRef}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files?.[0]; if (f) pickFile(f); }}
              className={`relative flex min-h-[260px] items-center justify-center overflow-hidden rounded-lg border border-dashed ${dragging ? "border-primary bg-secondary" : "border-border"}`}
            >
              {imgUrl ? (
                <>
                  <img ref={imgRef} src={imgUrl} className="block max-h-[420px] max-w-full" alt="preview" />
                  <canvas ref={canvasRef} className="pointer-events-none absolute" />
                </>
              ) : (
                <p className="text-sm text-muted-foreground">选择或拖入图片后预览</p>
              )}
            </div>
            {imgUrl && (
              <div className="mt-3 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">下方为带识别框的重建图片</p>
                <Button size="sm" variant="outline" onClick={savePng}><Save className="h-4 w-4" />保存重建 PNG</Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><ScanText className="h-4 w-4" />识别结果</CardTitle>
            <Badge>{lines.length} 行</Badge>
          </CardHeader>
          <CardContent className="space-y-2">
            {lines.length === 0 && <p className="text-sm text-muted-foreground">尚未识别</p>}
            {lines.map((l, i) => (
              <div key={i} className="rounded-lg border bg-muted/50 p-2.5">
                <div className="break-all text-sm">{l.text}</div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-secondary">
                  <div className="h-full bg-primary" style={{ width: `${Math.round((l.confidence || 0) * 100)}%` }} />
                </div>
                <div className="mt-1 text-right text-[11px] text-muted-foreground">{Math.round((l.confidence || 0) * 100)}%</div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
import { useCallback, useEffect, useState } from "react";
import { Upload, Loader2, Download, Trash2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { toast } from "@/lib/toast";
import { YoloService } from "@bindings/cpm_orc/internal/app";

const PRESETS = [
  { scale: "n", label: "YOLO26-n", size: "~6MB", desc: "最快" },
  { scale: "s", label: "YOLO26-s", size: "~22MB", desc: "平衡" },
  { scale: "m", label: "YOLO26-m", size: "~52MB", desc: "中等" },
  { scale: "l", label: "YOLO26-l", size: "~88MB", desc: "高精度" },
  { scale: "x", label: "YOLO26-x", size: "~130MB", desc: "最高精度" },
];

export default function YoloPage() {
  const [loaded, setLoaded] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [confThresh, setConfThresh] = useState(0.5);
  const [iouThresh, setIouThresh] = useState(0.45);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [imgDim, setImgDim] = useState({ w: 0, h: 0 });

  const refresh = useCallback(async () => {
    try {
      setLoaded(await YoloService.IsLoaded());
      setLocalModels(await YoloService.ListLocalModels());
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const loadModel = async (path: string) => {
    try {
      await YoloService.Load(path);
      setModelPath(path);
      setLoaded(true);
      toast("模型加载成功");
    } catch (e) { toast(String(e), true); }
  };

  const downloadModel = async (scale: string) => {
    setDownloading(scale);
    try {
      const path = await YoloService.DownloadYOLO26(scale);
      toast("下载完成: " + path);
      await refresh();
      await loadModel(path);
    } catch (e) { toast(String(e), true); }
    finally { setDownloading(null); }
  };

  const detect = async () => {
    if (!imgSrc) { toast("请先选择图片", true); return; }
    setBusy(true);
    setDetections([]);
    try {
      // Strip data URI prefix
      const b64 = imgSrc.includes(",") ? imgSrc.split(",")[1] : imgSrc;
      const dets = await YoloService.Detect(b64, confThresh, iouThresh);
      setDetections(dets || []);
      toast(`检测到 ${(dets || []).length} 个目标`);
    } catch (e) { toast(String(e), true); }
    finally { setBusy(false); }
  };

  const onFile = (file: File) => {
    const r = new FileReader();
    r.onload = () => setImgSrc(r.result as string);
    r.readAsDataURL(file);
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">YOLO 目标检测</h1>
          <p className="mt-1 text-sm text-muted-foreground">YOLO26 ONNX 推理，支持图片目标检测与识别。</p>
        </div>
        <Badge variant={loaded ? "success" : "secondary"}>{loaded ? "已加载" : "未加载"}</Badge>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-sm font-medium">模型</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Quick download */}
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <Button key={p.scale} size="sm" variant="outline" onClick={() => downloadModel(p.scale)} disabled={!!downloading}>
                {downloading === p.scale ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                {p.label} {p.size}
              </Button>
            ))}
          </div>
          {/* Local models */}
          {localModels.length > 0 && (
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">本地模型</Label>
              {localModels.map((m) => (
                <div key={m} className="flex items-center gap-2">
                  <Button size="sm" variant={modelPath === m ? "default" : "ghost"} onClick={() => loadModel(m)} className="flex-1 justify-start text-xs truncate">
                    {m.split("/").pop()}
                  </Button>
                </div>
              ))}
            </div>
          )}
          {/* Manual path */}
          <div className="flex items-center gap-2">
            <Input value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="或手动输入 .onnx 模型路径" className="flex-1" />
            <Button variant="outline" onClick={() => loadModel(modelPath)}>加载</Button>
          </div>
        </CardContent>
      </Card>

      {/* Detection */}
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-medium">输入图片</CardTitle>
            <div className="flex items-center gap-2">
              <label className="cursor-pointer">
                <Button asChild variant="outline" size="sm"><span><Upload className="h-4 w-4" />选择图片</span></Button>
                <input type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
              </label>
              <Button size="sm" onClick={detect} disabled={busy || !loaded || !imgSrc}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}检测
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {/* Thresholds */}
            <div className="flex gap-4">
              <div className="flex items-center gap-2">
                <Label className="text-xs w-20">置信度阈值</Label>
                <Input type="number" step="0.05" min="0" max="1" value={confThresh}
                  onChange={(e) => setConfThresh(parseFloat(e.target.value) || 0.5)} className="w-20" />
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-xs w-20">IoU 阈值</Label>
                <Input type="number" step="0.05" min="0" max="1" value={iouThresh}
                  onChange={(e) => setIouThresh(parseFloat(e.target.value) || 0.45)} className="w-20" />
              </div>
            </div>
            {/* Image preview */}
            {imgSrc && (
              <div className="relative rounded-lg border overflow-hidden">
                <img src={imgSrc} alt="input" className="max-h-[400px] w-full object-contain"
                  onLoad={(e) => setImgDim({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
                {/* Draw bounding boxes */}
                {detections.map((d, i) => (
                  <div key={i} className="absolute border-2 border-green-400 bg-green-400/10"
                    style={{
                      left: `${(d.box[0] / (imgDim.w || 1)) * 100}%`,
                      top: `${(d.box[1] / (imgDim.h || 1)) * 100}%`,
                      width: `${((d.box[2] - d.box[0]) / (imgDim.w || 1)) * 100}%`,
                      height: `${((d.box[3] - d.box[1]) / (imgDim.h || 1)) * 100}%`,
                    }}>
                    <span className="absolute -top-5 left-0 bg-green-400 text-black text-[10px] px-1 rounded">
                      {d.label || `cls${d.class}`} {(d.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
            {!imgSrc && <p className="text-sm text-muted-foreground">选择图片后点击「检测」</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-medium">检测结果</CardTitle>
            <Badge>{detections.length} 个目标</Badge>
          </CardHeader>
          <CardContent>
            {detections.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无结果</p>
            ) : (
              <div className="max-h-[400px] overflow-y-auto space-y-1">
                {detections.map((d, i) => (
                  <div key={i} className="flex items-center justify-between rounded border px-3 py-1.5 text-sm">
                    <span className="font-mono">{d.label || `class_${d.class}`}</span>
                    <span className="text-muted-foreground">{(d.score * 100).toFixed(1)}%</span>
                    <span className="text-xs text-muted-foreground font-mono">
                      [{d.box.map((v: number) => v.toFixed(0)).join(", ")}]
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

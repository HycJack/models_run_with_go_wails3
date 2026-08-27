import { useState, useEffect } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { Percent, ImagePlus, Check, Wrench, Copy, CornerDownLeft, Loader2, Sparkles, Square, X, Server } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { MathService, OllamaService, LlmService } from "@bindings/cpm_orc";

function renderLatex(latex: string): string {
  try {
    return katex.renderToString(latex, { throwOnError: false, displayMode: true, strict: false });
  } catch {
    return `<span class="text-destructive">渲染失败</span>`;
  }
}

export default function MathPage() {
  const [nl, setNl] = useState("");
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageMeta, setImageMeta] = useState("data:image/png;base64");
  const [latex, setLatex] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [checks, setChecks] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"nl" | "image" | "edit">("nl");
  const [backend, setBackend] = useState<"onnx" | "ollama">("ollama");
  const [ollamaModels, setOllamaModels] = useState<any[]>([]);
  const [ollamaOk, setOllamaOk] = useState(false);
  const [textModel, setTextModel] = useState("");
  const [visionModel, setVisionModel] = useState("");
  const [zoom, setZoom] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const ok = await OllamaService.Ping();
        setOllamaOk(ok);
        if (!ok) { toast("Ollama 未运行，已回退本机 ONNX", true); setBackend("onnx"); return; }
        const ms = await OllamaService.ListModels();
        setOllamaModels(ms);
        const vision = ms.find((m: any) => /minicpm-v|vl|vision/i.test(m.name));
        setVisionModel(vision ? vision.name : (ms.find((m: any) => /qwen/i.test(m.name))?.name ?? ms[0]?.name ?? ""));
        setTextModel(ms.find((m: any) => /minicpm5/i.test(m.name))?.name ?? ms[0]?.name ?? "");
      } catch (e) {
        setOllamaOk(false);
        setBackend("onnx");
      }
    })();
  }, []);

  const apply = async (l: string) => {
    setLatex(l);
    setResult(l);
    setChecks([]);
    try {
      const v: any = await MathService.Validate(l);
      setChecks(v.checks || []);
    } catch (e) { console.error(e); }
  };

  const toLatex = async (source: "nl" | "image") => {
    setBusy(true);
    try {
      const l = source === "nl"
        ? (backend === "ollama"
            ? await OllamaService.ToLatex(textModel, nl)
            : await MathService.ToLatex(nl))
        : (backend === "ollama"
            ? await OllamaService.VisionToLatex(visionModel, imageB64!)
            : await MathService.OcrToLatex(imageB64!));
      await apply(l);
      setMode("edit");
      toast("已生成 LaTeX");
    } catch (e) { toast(String(e), true); }
    finally { setBusy(false); }
  };

  const repair = async () => {
    try {
      const r: any = await MathService.Repair(latex);
      await apply(r.latex);
      toast(r.changed ? "已自动修复" : "无需修复");
    } catch (e) { toast(String(e), true); }
  };

  const copy = async () => {
    try { await MathService.CopyText(latex); toast("已复制"); }
    catch (e) { toast(String(e), true); }
  };

  const insert = async () => {
    try { await MathService.InsertAtCursor(latex); toast("已插入到光标处"); }
    catch (e) { toast(String(e), true); }
  };

  const pickImage = (file: File) => {
    const r = new FileReader();
    r.onload = () => {
      const dataUrl = r.result!.toString();
      const idx = dataUrl.indexOf(",");
      setImageMeta(dataUrl.slice(0, idx));
      setImageB64(dataUrl.slice(idx + 1));
      setMode("image");
      toast("图片已载入，点「图片→LaTeX」");
    };
    r.readAsDataURL(file);
  };

  const clearImage = () => {
    setImageB64(null);
    setImageMeta("data:image/png;base64");
    if (mode === "image") setMode("nl");
  };

  const stop = () => {
    if (backend === "ollama") OllamaService.Stop();
    else LlmService.Stop();
    toast("已停止生成");
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">公式助手</h1>
          <p className="mt-1 text-sm text-muted-foreground">自然语言 / 公式图片 → LaTeX，确定性校验与修复，KaTeX 实时预览。</p>
        </div>
        <Badge variant="success">本地 · Ollama / ONNX</Badge>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 py-3">
          <div className="flex items-center gap-1 rounded-lg border p-1">
            <button onClick={() => setBackend("ollama")}
              className={`rounded-md px-3 py-1 text-sm ${backend === "ollama" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}>
              <Server className="mr-1 inline h-3.5 w-3.5" />Ollama {ollamaOk ? "●" : "○"}
            </button>
            <button onClick={() => setBackend("onnx")}
              className={`rounded-md px-3 py-1 text-sm ${backend === "onnx" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}>
              本机 ONNX
            </button>
          </div>
          {backend === "ollama" && (
            <>
              {!ollamaOk && <span className="text-xs text-destructive">Ollama 未运行（需启动 ollama 服务）</span>}
              <select value={textModel} onChange={(e) => setTextModel(e.target.value)} title="文本/规划模型"
                className="h-9 rounded-md border border-input bg-muted px-2 text-sm">
                {ollamaModels.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
              </select>
              <select value={visionModel} onChange={(e) => setVisionModel(e.target.value)} title="视觉模型（图片→公式）"
                className="h-9 rounded-md border border-input bg-muted px-2 text-sm">
                {ollamaModels.filter((m) => /minicpm-v|vl|vision/i.test(m.name)).map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
              </select>
            </>
          )}
          <span className="text-xs text-muted-foreground">
            {backend === "ollama"
              ? "Ollama：文本用 MiniCPM5-1B，图片用 MiniCPM-V 视觉模型直接识别（TeXada 同款方案，无需 PP-OCR）"
              : "本机 ONNX：文本用已加载的对话模型，图片走 PP-OCR 提文字 → LLM 整理"}
          </span>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Sparkles className="h-4 w-4" />生成 LaTeX</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <Input placeholder="用自然语言描述公式，如：x 的平方加 y 的平方开根号" value={nl}
                onChange={(e) => { setNl(e.target.value); setMode("nl"); }} onKeyDown={(e) => e.key === "Enter" && toLatex("nl")} />
              <Button onClick={() => toLatex("nl")} disabled={busy || !nl.trim()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Percent className="h-4 w-4" />}生成
              </Button>
              {busy && <Button variant="ghost" onClick={stop}><Square className="h-4 w-4 fill-current" />停止</Button>}
            </div>
            <div className="flex items-center gap-2 overflow-hidden">
              <label className="shrink-0 cursor-pointer">
                <Button asChild variant="outline"><span><ImagePlus className="h-4 w-4" />公式图片</span></Button>
                <input type="file" accept="image/*" className="hidden"
                  onChange={(e) => e.target.files?.[0] && pickImage(e.target.files[0])} />
              </label>
              {imageB64 && (
                <>
                  <div className="relative shrink-0">
                    <img src={`${imageMeta},${imageB64}`} onClick={() => setZoom(true)}
                      className="max-h-20 max-w-32 cursor-zoom-in rounded-md object-contain" alt="formula" title="点击放大" />
                    <Button size="icon" variant="ghost" onClick={clearImage} title="清除图片"
                      className="absolute -right-2 -top-2 h-5 w-5">
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <Button variant="secondary" onClick={() => toLatex("image")} disabled={busy}>
                    <Percent className="h-4 w-4" />图片→LaTeX
                  </Button>
                  {busy && <Button variant="ghost" onClick={stop}><Square className="h-4 w-4 fill-current" />停止</Button>}
                </>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {backend === "ollama"
                ? "图片识别：MiniCPM-V 视觉模型直接看图输出公式（无需 OCR）"
                : "图片识别：先 PP-OCR 提文字，再交给模型整理成 LaTeX（适合印刷体公式）"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2"><Wrench className="h-4 w-4" />校验 / 修复</CardTitle>
            <div className="flex gap-2">
              <Button size="sm" onClick={repair}><Wrench className="h-4 w-4" />自动修复</Button>
              <Button size="sm" variant="outline" onClick={copy}><Copy className="h-4 w-4" />复制</Button>
              <Button size="sm" variant="outline" onClick={insert}><CornerDownLeft className="h-4 w-4" />插入光标</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea placeholder="在这里编辑 LaTeX，如：\frac{a}{b} + \sqrt{x^2+y^2}"
              value={mode === "edit" ? latex : ""} onChange={(e) => { setLatex(e.target.value); setResult(e.target.value); }} rows={3} />
            {checks.length > 0 && (
              <div className="space-y-1">
                {checks.map((c, i) => (
                  <div key={i} className={`flex items-center gap-2 text-xs ${c.ok ? "text-emerald-400" : "text-destructive"}`}>
                    {c.ok ? <Check className="h-3.5 w-3.5" /> : <Wrench className="h-3.5 w-3.5" />}{c.detail}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Percent className="h-4 w-4" />预览</CardTitle>
        </CardHeader>
        <CardContent>
          {result ? (
            <div className="overflow-x-auto rounded-lg border bg-muted p-5 text-center"
              dangerouslySetInnerHTML={{ __html: renderLatex(result) }} />
          ) : (
            <p className="text-sm text-muted-foreground">生成或编辑 LaTeX 后在此预览</p>
          )}
          <pre className="mt-3 whitespace-pre-wrap break-all rounded-lg border bg-muted p-3 font-mono text-xs">{latex}</pre>
        </CardContent>
      </Card>

      {zoom && imageB64 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
          onClick={() => setZoom(false)}>
          <img src={`${imageMeta},${imageB64}`} alt="公式图片（点击关闭）"
            className="max-h-full max-w-full rounded-lg object-contain shadow-2xl" />
          <Button size="icon" variant="ghost" className="absolute right-4 top-4 h-9 w-9 text-white"
            onClick={() => setZoom(false)}>
            <X className="h-5 w-5" />
          </Button>
        </div>
      )}
    </div>
  );
}
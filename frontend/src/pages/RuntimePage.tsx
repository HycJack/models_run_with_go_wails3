import { useCallback, useEffect, useState } from "react";
import { Download, Cpu, Settings2, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { toast } from "@/lib/toast";
import { Events } from "@wailsio/runtime";
import { RuntimeService } from "@bindings/cpm_orc/internal/app";

export default function RuntimePage() {
  const [rt, setRt] = useState<any>(null);
  const [root, setRoot] = useState("");
  const [proxy, setProxy] = useState("");
  const [coreml, setCoreml] = useState(false);
  const [dl, setDl] = useState(0);
  const [downloading, setDownloading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setRt(await RuntimeService.Status());
      const cfg: any = await RuntimeService.Config();
      setRoot(cfg?.modelRoot || "");
      setProxy(cfg?.proxy || "");
      setCoreml(cfg?.coreML || false);
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    const off = Events.On("dl:progress", (e: any) => {
      const d = e.data;
      if (d.total) setDl(Math.min(100, Math.round((d.done / d.total) * 100)));
    });
    const offDone = Events.On("dl:done", () => { setDownloading(false); setDl(0); refresh(); });
    return () => { off(); offDone(); };
  }, [refresh]);

  const download = async () => {
    setDownloading(true);
    setDl(0);
    try {
      await RuntimeService.Download("");
      toast("ONNX Runtime 已安装");
      await refresh();
    } catch (e) { toast(String(e), true); } finally { setDownloading(false); }
  };

  const save = async () => {
    try {
      const cfg: any = await RuntimeService.Config();
      if (!cfg) { toast("读取配置失败", true); return; }
      cfg.modelRoot = root;
      cfg.proxy = proxy;
      cfg.coreML = coreml;
      await RuntimeService.SaveConfig();
      toast("配置已保存");
    } catch (e) { toast(String(e), true); }
  };

  const testProxy = async () => {
    try { await RuntimeService.TestProxy(proxy); toast("代理可用"); }
    catch (e) { toast(String(e), true); }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold">运行环境</h1>
        <p className="mt-1 text-sm text-muted-foreground">ONNX Runtime 与全局配置。</p>
      </div>

      <div className="grid grid-cols-3 gap-5">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Cpu className="h-4 w-4" />ONNX Runtime</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">状态</span>
              <Badge variant={rt?.present ? "success" : "destructive"}>{rt?.present ? "已安装" : "未安装"}</Badge>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">版本</span><span className="font-mono text-xs">{rt?.version || "-"}</span>
            </div>
            <div className="text-xs text-muted-foreground break-all">库：{rt?.path || "-"}</div>
            <Button className="w-full" onClick={download} disabled={downloading}>
              <Download className="h-4 w-4" />{downloading ? "下载中…" : "下载 / 更新运行时"}
            </Button>
            {downloading && <Progress value={dl} />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Settings2 className="h-4 w-4" />全局设置</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1"><Label>模型根目录</Label><Input value={root} onChange={(e) => setRoot(e.target.value)} /></div>
            <div className="space-y-1">
              <Label>网络代理</Label>
              <div className="flex gap-2">
                <Input value={proxy} onChange={(e) => setProxy(e.target.value)} placeholder="http://127.0.0.1:7890" />
                <Button variant="outline" onClick={testProxy}>测试</Button>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={coreml} onCheckedChange={setCoreml} />
              <Label>CoreML EP（OCR 加速，实测更慢，默认关）</Label>
            </div>
            <Button className="w-full" onClick={save}>保存配置</Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Info className="h-4 w-4" />关于</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5 text-sm text-muted-foreground">
            <p className="flex justify-between"><span>框架</span><code className="text-xs text-primary">Go + Wails3</code></p>
            <p className="flex justify-between"><span>推理引擎</span><code className="text-xs text-primary">ONNX Runtime</code></p>
            <p className="flex justify-between"><span>语音</span><code className="text-xs text-primary">FunASR / whisper.cpp</code></p>
            <p className="flex justify-between"><span>对话</span><code className="text-xs text-primary">MiniCPM5 / Qwen</code></p>
            <p className="pt-2 text-xs">
              模型：PP-OCRv6 三档 + 方向分类 · FunASR SenseVoiceSmall · whisper · MiniCPM5-1B / Qwen3-0.6B / Qwen2.5-0.5B
            </p>
            <p className="text-xs">资源：HuggingFace Hub（模型）· GitHub Releases（运行时）· whisper.cpp（Metal）</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
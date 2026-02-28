# audio2text-server

Build:

```shell
# Download models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2
tar xvf sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2
mv sherpa-onnx-zipformer-vi-30M-int8-2026-02-09 models/
rm sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2

docker build -t asr_bot:py310 .
```

Run it

```shell
docker run -it --rm -m 16g -v "$(pwd):/app" -p 8000:8000 --name asr_bot asr_bot:py310 bash
```

# AIC 2026 — research and validation report

## Evidence status (reviewed 2026-08-20)

The official gate is **BLOCKED**, not a claimed `>90%`: this public checkout
contains no organizer query/ground-truth/scorer bundle. Any metadata-derived
proxy is quarantined because the same metadata can appear in the retrieval
index. The reproducible public evidence is therefore limited to code/schema
smoke tests and the documented local experiment summaries.

The updated CSV/ZIP upload contract is recorded in
[`AIC26_OFFICIAL_UPDATE_20260820.md`](AIC26_OFFICIAL_UPDATE_20260820.md).

## Kết luận ngắn

Chưa có cơ sở trung thực để cam kết accuracy trên 90% cho đề thật: workspace
không chứa query/ground-truth chính thức, còn benchmark hiện tại chủ yếu lấy
video metadata làm pseudo-label và bỏ qua frame interval, QA answer và TRAKE
partial R-Score.

The data-bearing local checkout has a separate statement/data audit; this
public checkout intentionally contains no videos, indexes, query, GT, or
organizer scorer. `eval_official.py` is therefore only an official-style
compatibility evaluator and must not be used to claim competition accuracy.

Kiểm tra lại website chính thức ngày 2026-08-20: trang chủ mô tả AI Challenge
2026 là trợ lý truy xuất multimedia, nói sẽ thử nghiệm cả hình thức tự động và
khuyến khích LVLM/GenAI,
nhưng chỉ công bố lịch dự kiến và chưa công bố query/GT/scorer của vòng tự động
([trang chính thức](https://aichallenge.hochiminhcity.gov.vn/)). Trang hướng
dẫn công khai hiện vẫn là tài liệu AI Challenge 2022, nên không được dùng làm
ground truth cho AIC2026.

Điểm đo được hiện tại:

| Tập proxy | R@1 | R@5 | R@20 | R@50 | R@100 | FinalScore |
|---|---:|---:|---:|---:|---:|---:|
| title + BM25 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | **0.990** |
| description + BM25 | 0.460 | 0.545 | 0.660 | 0.725 | 0.785 | **0.635** |
| visual-only mCLIP, isolated 8-frame/video test | 0.000 | — | 0.160 | 0.295 | 0.480 | **0.195** |
| visual-only SigLIP2, isolated 8-frame/video test | 0.045 | 0.085 | 0.260 | 0.420 | 0.545 | **0.271** |
| visual-only SigLIP2, full 177,321-keyframe test | 0.070 | 0.140 | 0.260 | 0.375 | 0.480 | **0.265** |

Các benchmark title/description ở trên vẫn là metadata video-level pseudo-GT
và phải giữ nhãn `PROXY`/`QUARANTINED_LEAKAGE`; chúng không xác nhận accuracy.
H1/H2/H3 trên manual fixture leak-free được ghi riêng trong
[`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md), nhưng chỉ có 10 query và
không có scorer parity.

Title score cao không đại diện cho visual retrieval: query được lấy trực tiếp
từ title và title cũng nằm trong BM25 documents. Không nên dùng con số này để
suy ra điểm thi thật.

## Các nhánh đã thử

1. BM25, CLIP-B/32, mCLIP và RRF/objects: giữ BM25 cho title-like query;
   mCLIP là nhánh dense đáng giữ cho description tiếng Việt.
2. SigLIP-base: checkpoint đã có cache nhưng bản thử nghiệm không multilingual;
   full encode bị dừng vì GPU contention và artifact chỉ có 256/6,984 vector
   khác 0. Không tích hợp artifact này vào production.
3. SigLIP2-base-patch16-224: đã encode đủ 177.321/177.321 keyframe, dựng
   FAISS 768-dim đầy đủ và benchmark description proxy được
   `R@1=0.070`, `R@100=0.480`, `FinalScore=0.265`. Bản full không vượt bản
   sample 8-frame/video và thấp hơn BM25 metadata proxy. Thử fusion min-max
   với BM25 trên cùng 200 description query cho `FinalScore=0.603` ở
   SigLIP weight 0, giảm còn `0.579` ở weight 0.15; vì vậy SigLIP2 giữ ở
   dạng opt-in, không bật mặc định.
4. Temporal smoothing: làm mCLIP description proxy giảm FinalScore
   `0.195 -> 0.175`; không bật mặc định cho KIS/Q&A, chỉ nên dùng trong
   candidate video và TRAKE.
5. Paper-derived directions: SigLIP2 multilingual, CLIP4Clip/X-CLIP-style
   temporal aggregation, Moment-DETR-style ordered event scoring, và
   multi-frame Qwen2.5-VL cho QA. Các hướng này cần ground truth thật trước
   khi chọn trọng số.

Hai paper AIC'25 mới được rà lại củng cố lựa chọn này: MERVIN dùng keyframe,
transcript và video summary, đạt 79/88 ở vòng loại; U-CESE dùng caption theo
shot, hợp nhất các modality và temporal consistency. Đây là kết quả cùng họ
dữ liệu, nhưng vẫn không phải validation của AIC2026.

Rà soát model bổ sung cho tiếng Việt cho thấy multilingual CLIP hiện là lựa
chọn chạy ngay và đã được smoke-test offline. SigLIP2-base-patch16-224 đã
được tải hoàn chỉnh và benchmark trong thư mục isolated; chưa đưa vector
SigLIP2 vào production vì proxy visual vẫn thấp hơn BM25 metadata. MEXMA-SigLIP2
và ViSigLIP-OT có tín hiệu tốt hơn cho tiếng Việt nhưng chưa được cache/kiểm
chứng trong workspace. Không dùng các checkpoint gated hay native video
encoder chưa có benchmark tiếng Việt để suy diễn accuracy.

## ASR branch đã triển khai thử

Trong checkout public, dữ liệu không được commit. Code có [asr_index.py](../solution/asr_index.py): PhoWhisper đọc audio qua
ffmpeg, lưu đoạn có timestamp, gom thành cửa sổ khoảng 15 giây, rồi search
BM25 và ánh xạ timestamp về `frame_idx` qua CSV. `EnsembleRetriever` ưu tiên
`ensemble_index/asr_full.jsonl` và fallback về `asr_index.jsonl` cho checkout cũ.

Các artifact full-corpus được tạo trong checkout data-bearing, không được đưa
vào public. Batch1024 đã thử nhưng chạm sát VRAM trên máy chia sẻ; batch64 là
cấu hình an toàn hơn. Với long-form
`ChunkPipeline`, Transformers tự giới hạn `num_workers` về 1 để tránh lỗi dù
được truyền `--workers 8`; đây là giới hạn của pipeline, không phải bỏ sót
tham số.

OCR cũng là một nhánh tùy chọn; `EnsembleRetriever` ưu tiên
`ensemble_index/ocr_full.jsonl` và fallback
về `ocr_index.jsonl`; các JSONL runtime này không được commit lên GitHub.
Đã tối ưu script sang `readtext_batched()` và kiểm tra runtime trên 2 ảnh
(`ocr_batched=PASS`) để tránh detector tuần tự từng ảnh.

Smoke test trên một video có lời thoại cho transcript tiếng Việt có nội dung
đúng về trạm cứu hộ mèo và địa danh; clip nhạc gây lặp với checkpoint tiny.
Đây là kiểm tra plumbing, không phải accuracy benchmark. SigLIP2 được encode
vào artifact isolated `experiments/siglip_temporal/artifacts/`; kết quả không
sửa index production.

Qwen2.5-VL-3B được gọi qua pipeline QA nhưng cần GPU có headroom an toàn.
`submission_ens.py`
đã được làm cho graceful: bỏ CLIP phụ khi bật VLM và tiếp tục KIS/TRAKE nếu
VLM không nạp được; muốn chấm QA thật cần chạy trên GPU trống.

## Paper sát đề nhất

`Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval`
([arXiv](https://arxiv.org/abs/2606.19682), 2026) mô tả một hệ thống cho
AIC'25 với đúng kiểu dữ liệu keyframe/object/CLIP và đúng cách tính R-Score
năm operating points. Pipeline của paper kết hợp:

- Qwen2.5-VL để tạo caption/OCR và Whisper để lập chỉ mục ASR có timestamp;
- CLIP và SigLIP2 để lấy embedding bổ sung cho nhau;
- Reciprocal Rank Fusion (RRF), sau đó rerank theo temporal context;
- nhiều frame lân cận và Rocchio feedback cho các lượt truy vấn sau.

Paper báo cáo 79.6/88 trên preliminary AIC'25, tương đương khoảng 90.5%.
Đây là bằng chứng thực nghiệm rằng mốc 90% có thể đạt được trên dữ liệu cùng
họ, nhưng không phải ground truth của AIC2026 nên chưa thể chuyển thành cam
kết cho bộ test hiện tại. Đây là hướng ưu tiên số 1 nếu có GPU/thời gian để
caption/ASR toàn bộ 873 video.

Nguồn sơ cấp đã kiểm chứng và bản đồ transferability chi tiết nằm trong
[`PAPER_MAP.md`](PAPER_MAP.md). MERVIN và Vortex báo cáo kết quả trên AIC HCMC
2025; DANTE mô tả dynamic programming cho TRAKE. Không bài nào cung cấp bằng
chứng scorer-parity trên AIC26.

Khảo sát thêm SOICT 2025 cho thấy các nhóm đạt điểm cao khác cũng dùng cùng
pattern: TARS tách query thành sub-event rồi monotonic DP/K-pointer (báo cáo
93.15% Top-1); các pipeline VLM + temporal algorithm báo cáo 97%; Perception
Browser báo cáo 84.4/88 (~95.91%). Các số này là preliminary/team-reported với
mẫu số khác nhau, không phải leaderboard AIC2026, nhưng củng cố việc ưu tiên
hybrid modality và temporal alignment thay vì chỉ tăng kích thước CLIP.

## Sửa lỗi đã áp dụng

- thêm `import torch` cho decorator `@torch.no_grad()` trong v8, v9 và TRAKE v2;
- sửa resolver Keyframes L26 (`Keyframes_L26_a` ... `_e`) cho SigLIP re-encode,
  Qwen VLM và QA solver;
- ensemble giữ query tiếng Việt cho BM25 thay vì truyền bản dịch tiếng Anh;
- solver v8 load `per_video.pkl` một lần thay vì unpickle lại theo từng video;
- ensemble padding deterministic thay cho random;
- sửa evaluator ensemble stale API;
- thêm nhánh ASR tùy chọn và không bật nếu chưa có transcript index;
- TRAKE trong `submission_ens --use-ensemble` dùng candidate fusion + event/frame
  monotonic DP thay vì tạo CLIP retriever thứ hai và bỏ qua ensemble;
- thêm OCR branch tùy chọn và frame-level fallback theo ASR/OCR timestamp;
- thêm `eval_official.py` để chấm đúng năm operating points
  `R@1,R@5,R@20,R@50,R@100` và điểm frame/answer/TRAKE khi có GT.
- sửa parser evaluator để QA answer dạng số (ví dụ `5`) không bị hiểu nhầm
  thành frame thứ hai; helper tests KIS/QA/TRAKE đã PASS.
- siết schema evaluator: kiểm tra loại query, arity, frame không âm, rank liên
  tục, query-id trùng/không biết và GT thiếu khoảng/event; không còn fail-open
  khi dữ liệu đánh giá bị thiếu.
- sửa TRAKE để dùng monotonic DP trên toàn bộ candidate video, giữ backup
  video ở các operating point sớm và phạt mềm các chuỗi event trải quá rộng;
  smoke test synthetic đạt 4/4 event trong range.
- thêm dense source-frame refinement quanh các keyframe tốt nhất để giảm lỗi
  khoảng cách lớn giữa keyframe và frame gốc; đây là nhánh query-time, không
  làm thay đổi production FAISS.
- deduplicate object detections theo `(video, frame, class)` trước khi tạo
  inverted index và sửa SigLIP ordinal về `frame_idx` chính thức qua CSV map.

## Paper tham khảo

- CLIP: <https://arxiv.org/abs/2103.00020>
- SigLIP: <https://arxiv.org/abs/2303.15343>
- SigLIP 2 (multilingual): <https://arxiv.org/abs/2502.14786>
- mCLIP: <https://aclanthology.org/2023.acl-long.728/>
- CLIP4Clip: <https://arxiv.org/abs/2104.08860>
- X-CLIP: <https://arxiv.org/abs/2207.07285>
- VideoCLIP: <https://arxiv.org/abs/2109.14084>
- Temporal sentence grounding survey: <https://arxiv.org/abs/2201.08071>
- MERVIN (AIC'25, transcript + summary + visual): <https://arxiv.org/abs/2605.16120>
- U-CESE (AIC'25, unified multimodal/temporal search): <https://arxiv.org/abs/2605.23274>
- Vortex (AIC'25 multi-modal fusion): <https://arxiv.org/abs/2606.19682>
- TARS/giải pháp AIC'25 và các kết quả preliminary: [SOICT 2025 Program Book](https://soict.org/wp-content/uploads/2025/12/SOICT2025-ProgramBook.pdf)
- PhoWhisper Vietnamese ASR: [model](https://huggingface.co/vinai/PhoWhisper-small), [paper](https://research.vinai.io/wp-content/uploads/2024/05/239_phowhisper_automatic_speech_re.pdf)
- Qwen2.5-VL model card (multi-image/video):
  <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

## Chạy khi có query/GT thật

```bash
cd <checkout>/solution
CUDA_VISIBLE_DEVICES=0 python3 solver_v7.py \
  --queries /path/to/queries.jsonl --out /path/to/submission.jsonl
python3 eval_official.py --pred /path/to/submission.jsonl \
  --gt /path/to/ground_truth.jsonl
```

Chỉ kết luận “>90%” sau khi `eval_official.py` đạt ít nhất `FinalScore > 0.90`
trên tập nhãn thật, hoặc trên một validation split có frame intervals/answers
tương thích hoàn toàn với đề.

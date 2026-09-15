# MOV3: Real-CLIP visualization of common-anchor residual geometry

## Mục tiêu

MOV3 là một **motivation visualization**, không phải thí nghiệm đánh giá chất
lượng unlearning. Workflow này dùng embedding thật của CLIP để minh họa hai ý:

1. cùng dùng một anchor `person` không làm cho residual của nhiều target tự
   động nằm trong một không gian con hạng thấp;
2. `target_global_pairwise_residual_subspace` (TGPRS) xây dựng residual trong
   một pairwise subspace có hạng được kiểm soát, ở đây là \(k=30\).

MOV3 không train checkpoint, không sinh ảnh diffusion và không chạy GCD/FID.

## Cấu hình cố định

Cấu hình nằm trong `config.yaml`:

```yaml
model_id: CompVis/stable-diffusion-v1-4
benchmark_csv: data/100_celebrity.csv
anchor_prompt: person
target_count: 100
residual_rank: 30
embedding_batch_size: 32
```

- Target prompts: 100 tên celebrity lấy nguyên văn từ cột `concept` của các
  dòng có `type=erase` trong `data/100_celebrity.csv`.
- Anchor prompt: đúng một từ `person` cho tất cả target.
- Text encoder: CLIP text encoder đi kèm Stable Diffusion v1.4.
- Số chiều embedding: \(d=768\).
- Không thêm prompt template như `a photo of ...`.

## Trích xuất CLIP embedding

Mỗi prompt được tokenize với padding/truncation theo
`tokenizer.model_max_length`. Từ `last_hidden_state`, code lấy hidden state tại
token nội dung cuối cùng, tức vị trí ngay trước token EOS:

```python
subject_index = attention_mask.sum() - 2
embedding = last_hidden_state[subject_index]
```

Vì vậy mỗi celebrity name và anchor `person` được biểu diễn bằng đúng một vector
768 chiều. Ký hiệu:

\[
t_i\in\mathbb{R}^{768},\qquad a\in\mathbb{R}^{768},\qquad i=1,\ldots,100.
\]

## Common-anchor residual

Baseline dùng cùng anchor cho mọi target:

\[
r_i^{\mathrm{common}}=a-t_i.
\]

Ghép 100 residual theo hàng tạo thành:

\[
R_{\mathrm{common}}
=
\begin{bmatrix}
(a-t_1)^\top\\
\vdots\\
(a-t_{100})^\top
\end{bmatrix}
\in\mathbb{R}^{100\times768}.
\]

Mặc dù cùng kết thúc tại `person`, các residual vẫn phụ thuộc vào \(t_i\), do đó
không có lý do để chúng cùng hướng hoặc có hạng thấp.

## TGPRS residual

### 1. Xây global pairwise residual matrix

Với mỗi source target \(t_i\), workflow tạo residual tới tất cả target còn lại
và tới fixed anchor `person`:

\[
\mathcal G_i
=
\{t_j-t_i\mid j\ne i\}
\cup
\{a-t_i\}.
\]

Với 100 targets và một extra anchor, mỗi source có \(99+1=100\) directions.
Global matrix vì vậy có:

\[
100\times100=10{,}000
\]

residual rows trong không gian 768 chiều.

### 2. Học pairwise subspace rank 30

Mỗi hàng của global matrix được chuẩn hóa về unit norm trước SVD. Nếu

\[
\widehat G=U\Sigma V^\top,
\]

thì 30 right singular vectors đầu tạo basis:

\[
B=V_{1:30}^\top\in\mathbb{R}^{30\times768}.
\]

Phép chiếu lên subspace là:

\[
P_B(x)=xB^\top B.
\]

### 3. Tạo một residual riêng cho mỗi target

Implementation hiện tại chọn hướng ngược với projection của target và khôi
phục norm của common-anchor residual:

\[
\widetilde r_i
=
-\lVert a-t_i\rVert_2
\frac{P_B(t_i)}{\lVert P_B(t_i)\rVert_2}.
\]

Nếu \(P_B(t_i)\) gần zero, code lần lượt fallback sang projected
common-anchor residual rồi basis vector đầu tiên để tránh NaN. Vì mọi
\(\widetilde r_i\) đều thuộc span của \(B\):

\[
\operatorname{rank}(R_{\mathrm{TGPRS}})\le 30.
\]

Effective anchor riêng của target \(i\) được định nghĩa là:

\[
\widetilde a_i=t_i+\widetilde r_i.
\]

Các effective anchors này không phải một shared center.

## PCA dùng trong visualization

PCA được fit một lần trên 101 vector gồm 100 target embeddings và anchor
`person`. Dữ liệu được mean-center trước SVD. Hai principal components đầu giải
thích lần lượt:

- PC1: 5.56% variance;
- PC2: 3.78% variance.

Tổng cộng hai trục chỉ giữ khoảng 9.34% variance. PCA chỉ phục vụ visualization;
mọi phép tính rank được thực hiện trong không gian CLIP 768 chiều.

TGPRS residual được chiếu vào đúng hai PCA components:

\[
\widetilde r_i^{(2D)}=\widetilde r_iV_{1:2}^\top.
\]

## Cách tạo từng figure

Ba PDF được tạo **native từ ba Matplotlib canvas độc lập** trong
`native_panels.py`. Workflow không tạo một PDF tổng rồi crop.

### `common_anchor_residuals.pdf`

- vòng tròn xanh: 100 target embeddings trong PCA 2D;
- ngôi sao cam: embedding của `person`;
- mỗi đoạn xanh nối trực tiếp \(t_i\) tới common anchor \(a\);
- không hiển thị tên celebrity, title, legend, ticks hoặc axis labels;
- chữ duy nhất trong PDF là `"person"`.

Figure này minh họa rằng một shared endpoint vẫn tạo ra nhiều residual
directions khác nhau.

### `target_specific_residual_pairs.pdf`

Figure này ưu tiên biểu diễn từng cặp riêng mà không tạo ảo giác các vector hội
tụ vào một tâm:

- vòng tròn xanh: target \(t_i\);
- chấm cam: effective anchor \(\widetilde a_i\);
- mũi tên cam: \(\widetilde r_i\), nối \(t_i\) tới \(\widetilde a_i\);
- đủ 100 residual vectors được hiển thị;
- không có text, ticks hoặc axis labels.

Để tránh 100 đoạn thẳng giao nhau trong absolute PCA coordinates, mỗi cặp được
tịnh tiến sang một ô riêng trên lưới \(10\times10\). Grid position không mang ý
nghĩa embedding. Hướng của vector vẫn là hướng PCA của
\(\widetilde r_i^{(2D)}\). Một global display scale được tính từ percentile 90
của residual length; vector quá dài được clip để nằm trong ô. Do đó figure giữ
hướng và độ dài tương đối ở phần lớn mẫu, nhưng không nên dùng để đọc khoảng
cách CLIP tuyệt đối.

### `residual_rank_spectrum.pdf`

Code tính toàn bộ singular values của hai residual matrices
\(R_{\mathrm{common}}\) và \(R_{\mathrm{TGPRS}}\), sau đó vẽ:

\[
\sigma_j/\sigma_1.
\]

Numerical-rank threshold là:

\[
\tau=\max(N,d)\,\epsilon_{\mathrm{float32}}.
\]

Đường chấm ngang biểu diễn \(\tau\), còn đường dọc đánh dấu \(k=30\). Giá trị
hiển thị được floor tại \(10^{-8}\) để vẽ log scale; numerical rank luôn được
tính từ singular values gốc, trước khi floor. Minor ticks của log-scale bị tắt
để giảm nhiễu thị giác.

## Kết quả hiện tại

Từ `outputs/summary.json`:

| Quantity | Common anchor | TGPRS |
|---|---:|---:|
| Residual numerical rank | 100 | 30 |
| Stable rank | 1.689 | 5.168 |
| Spectral effective rank | 11.546 | 23.209 |
| Edit-statistic numerical rank | 100 | 30 |

Edit statistic được tính bằng:

\[
D=\frac{1}{N}R^\top T,
\]

với \(T\in\mathbb{R}^{N\times768}\). Kết quả phù hợp với liên hệ:

\[
R\rightarrow D=R^\top T/N\rightarrow\Delta W,
\qquad
\operatorname{rank}(\Delta W)
\le\operatorname{rank}(D)
\le\operatorname{rank}(R).
\]

MOV3 chứng minh trực tiếp sự khác biệt về geometry/rank của residual
construction. Nó không tự chứng minh rằng rank thấp cải thiện erasure quality,
retention hoặc optimization; các claim đó cần evaluation riêng.

## Chạy lại toàn bộ trên Vast AI

```bash
ssh -p 40234 root@182.224.239.168 -L 8080:localhost:8080
cd /workspace/CEdit
PYTHON_BIN=/venv/main/bin/python bash motivation/mov3/run.sh
```

Lần chạy này load Stable Diffusion v1.4 tokenizer/text encoder, tính lại CLIP
embeddings, residuals, PCA, spectrum và toàn bộ outputs.

## Chỉ render lại từ cached artifacts

Không cần GPU và không load lại CLIP:

```bash
cd /workspace/CEdit
/venv/main/bin/python motivation/mov3/render_cached.py \
  --output-dir motivation/mov3/outputs
```

Cached renderer đọc:

- `outputs/embedding_projection.csv`;
- `outputs/residual_spectrum.csv`;
- `outputs/summary.json`.

## Outputs

```text
motivation/mov3/outputs/
├── common_anchor_residuals.pdf
├── target_specific_residual_pairs.pdf
├── residual_rank_spectrum.pdf
├── common_anchor_clip_geometry.png
├── embedding_projection.csv
├── residual_spectrum.csv
└── summary.json
```

Ba PDF là vector outputs độc lập. PNG là preview tổng hợp và không được dùng để
tạo/crop các PDF.

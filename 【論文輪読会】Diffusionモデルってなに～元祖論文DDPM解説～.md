今回のコードは<a href="https://colab.research.google.com/drive/1FQ0vUoDtZ539_n37bc49srVT0Jr8-eOy?usp=sharing">【こちら】</a>

Denoising Diffusion Probabilistic Models<a href="https://arxiv.org/pdf/2006.11239.pdf">【こちら】</a>

## ひとことまとめ

- GANと比べて、一つのモデルしかない
- 情報からノイズ、ノイズから情報へ
- 学習過程（FP）と生成過程（RP）はマルコフ連鎖で決まれ
- 数学的完備

<div align="center">
 <img src="/upload/2023/06/img230608-9.png" width="1000px">
   <p>
  <font size="2" color="gray">図0</font>
 </p>
</div>

## Forword Process

実画像にノイズを加えていく、その過程はマルコフ連鎖で定義されている。

<div align="center">
 <img src="/upload/2023/06/img230608-1.png" width="1000px">
   <p>
  <font size="2" color="gray">図1</font>
 </p>
</div>

  
$$\begin{cases}
q_\theta(X_{1:T}|X_0) := ∏q(X_t|X_{t-1})\\
q_\theta(X_t|X_{t-1}) := N(X_t;\sqrt{1-\beta_t}X_{t-1},\beta_t I)
\end{cases}\tag{1}$$


ここでの$\beta_t$は加えたノイズの強さと考えてよい

二つのガウス分布$N(0,\sigma ^2_1 I)$ $N(0,\sigma ^2_2 I)$を足し合わせると新しいガウス分布$N(0,(\sigma ^2_1 + \sigma ^2_2) I)$になることにより

  
$$a_t = 1 - \beta_t $$
$$X_t\begin{cases}
=\sqrt{a_t} X_{t-1}+\sqrt{1- a_t} \epsilon_{t-1}\\
=\sqrt{a_t}(\sqrt{a_{t-1}} X_{t-2}+\sqrt{1- a_{t-1}} \epsilon_{t-2})+\sqrt{1- a_{t-1}} \epsilon_{t-1}\\
=\sqrt{a_ta_{t-1}} X_{t-2}+\sqrt{1- a_ta_{t-1}} \overline \epsilon_{t-2}\\
=...\\
=\sqrt{\overline a_t}X_0+\sqrt{1-\overline a_t}\epsilon
\end{cases}\tag{2}$$

つまり


$$q(X_t|X_0) = N(X_t;\sqrt{\overline a_t}X_{0},(1-\overline a_t) I)$$

<div align="center">
 <img src="/upload/2023/06/img230608-2.png" width="1000px">
   <p>
  <font size="2" color="gray">図2</font>
 </p>
</div>

よって、$t$時の画像の状態は$X_0$, $t$だけに依存する
（あとスケールを決めるハイパラだけ）

これで何がうれしいか？
- 時系列ではなく独立でBatch学習ができる

## Reverse Process

途中時刻ｔの画像のノイズを予測し、ノイズを抜いていく過程。
これもマルコフ連鎖で定義されている。

$$X_{t-1} \rightarrow X_t -noise$$

$$\begin{cases}
q_\theta(X_T)=N(X_t;0;I)\\
q_\theta(X_{0:T}|X_0) := q(x_T)∏q_\theta(X_{t-1}|X_{t})\\
q_\theta(X_{t-1}|X_{t}) := N(X_{t-1};  \mu _\theta (X_{t},t), \Sigma(X_t,t))
\end{cases}\tag{3}$$

      
- $\mu _\theta (X_{t},t)$でネットワークを使う
++可能条件について厳密な証明はあるが、証明が難しいため一旦とばす。（DDPM P12-A）++

### Choose a network

<div align="center">
   <img src="/upload/2023/06/img230608-8.jpg" width="1000px">
   <p>
  <font size="2" color="gray">図3-1</font>
 </p>
</div>

UNet!です
- 入力：
$t$時刻の画像
- 出力：
$t$時刻の画像のノイズ
- 損失/目的関数
予測ノイズと実際ノイズのL2正則
++厳密な証明はあるが一旦とばす。（DDPM P12-A）++


しかしこれだけじゃ足りない！
ノイズの強さは時刻によるものなので、時刻のembeddingが必要。

<div align="center">
 <img src="/upload/2023/06/img230608-5.png" width="1000px">
   <p>
  <font size="2" color="gray">図4</font>
 </p>
</div>

- ここで使ったembeddingはposition embedding
- transformerでも使われている

<div align="center">
 <img src="/upload/2023/06/img230608-6.png" width="1000px">
   <p>
  <font size="2" color="gray">図5</font>
 </p>
</div>

$k$　～　位置情報
$i$　～　i番目のパラメータ
$d$　～　dimension
$n$　～　定数・Attention is all you needでは10000

<div align="center">
 <img src="/upload/2023/06/img230608-7.jpg" width="500px">
   <p>
  <font size="2" color="gray">図6</font>
 </p>
</div>



## 流れのまとめ

<div align="center">
 <img src="/upload/2023/06/img230608-4.png" width="1000px">
   <p>
  <font size="2" color="gray">図7</font>
 </p>
</div>

## 実際まわしてみよう！

今回のコードは<a href="https://colab.research.google.com/drive/1FQ0vUoDtZ539_n37bc49srVT0Jr8-eOy?usp=sharing">【こちら】</a>

## 参考
Understanding Diffusion Models: A Unified Perspective: https://arxiv.org/pdf/2208.11970.pdf
Diffusion models from scratch in PyTorch: https://www.youtube.com/watch?v=a4Yfz2FxXiY
Tutorial on Denoising Diffusion-based Generative Modeling: Foundations and Applications: https://www.youtube.com/watch?v=cS6JQpEY9cs
Positional Embeddings: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1

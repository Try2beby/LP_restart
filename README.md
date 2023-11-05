# Implementation

## Problems

原问题

$$
\min_{x\ge 0}\max_{y} {\mathcal L}(x,y) = c^\top x+y^\top b-y^\top Ax.
$$

$$
F(z)= \begin{pmatrix}
 \nabla_x {\cal L}(x,y)\\
 -\nabla_y {\cal L}(x,y)
\end{pmatrix}
=\begin{pmatrix}
 c-A^\top y\\
 Ax-b
\end{pmatrix}
$$

### PrimalDualStep

- [ ] PPM 解

<p align="center">
<img src="assets/image-20230918192334096.png" alt="image-20230918192334096.png" width="50%">
</p>

- [ ] EGM 解二次优化问题

<p align="center">
<img src="assets/image-20230918192429086.png" alt="image-20230918192429086.png" width="50%">
</p>

迭代公式：

$$
\begin{align*}
& \hat z^{t+1} ={\rm Proj}_Z ( z^t-\eta F(z^t) )\\
& z^{t+1}={\rm Proj}_Z(z^t-\eta F(\hat z ^{t+1}))
\end{align*}
$$

- [ ] PDHG 解二次优化问题

<p align="center">
<img src="assets/image-20230918192640621.png" alt="image-20230918192640621.png" width="50%">
</p>

迭代公式

$$
\begin{align*}
& x^{t+1}=(x^t -\eta /w (c-A^\top y))_+\\
& y^{t+1}=y-\eta w (-b+A(2x^{t+1}-x^t))
\end{align*}
$$

- [ ] ADMM

<p align="center">
<img src="assets/image-20230929202435675.png" alt="image-20230929202435675.png" width="50%">
</p>

$$
\begin{align*}
&\theta_1=0, \theta_2(x_V)=c^\top x_V\\
&U=I,V=-I,q=0\\
&x_U^{t+1}=A^\top(AA^\top)^{-1}(b+A(-x_V^t-\frac1\eta y^t )) -(-x_V^t-\frac1\eta y^t ) \\
&x_V^{t+1}=x_U-\frac 1\eta y^t-\frac 1\eta c
\end{align*}
$$

<p align="center">
<img src="assets/image-20230929202527319.png" alt="image-20230929202527319.png" width="50%">
</p>

### normalized duality gap

$$
\begin{align*}
& \rho_r(z):=\frac{\max_{\hat{z}\in W_r(z)  }\{{\cal L}(x,\hat y)- {\cal L}(\hat x,y)\}}{r}\\
& W_r(z):=\{\hat z \in Z\mid \|z-\hat z \|\le r \}
\end{align*}
$$

若

$$
{\cal L}(x,y)=c^\top x+y^\top b-y^\top Ax,\quad x\ge 0,y\in {\mathbb R}^m
$$

求在 $z$ 处的 $\rho(z)$ 即要求一个二次约束线性目标函数的优化问题

<p align="center">
<img src="assets/image-20230918191501489.png" alt="image-20230918191501489.png" width="50%">
</p>

具体地

$$
\begin{align*}
{\cal L}(x,\hat y)-{\cal L}(\hat x,y)&=c^\top x+\hat y ^\top b-\hat y ^\top Ax -(c^\top \hat x+y^\top b -y^\top A \hat x)\\
&=\hat y ^\top (b-Ax) +(y^\top A-c^\top)\hat x +c^\top x -y^\top b
\end{align*}
$$

- 当 $Z={\mathbb R}^{m+n}$ 时，

$$
\rho_r(z)=\|F(z)\|=\left \|
\begin{pmatrix}
\nabla_x{\cal L}(x,y)\\
-\nabla _y {\cal L}(x,y)\\
\end{pmatrix}\right \|
$$

- 而对标准线性规划 $Z=X\times Y,X={\mathbb R}^m_+$ ，则需使用数值算法求值

可用算法如下：
![algo](assets/2023-10-23-18-14-49.png)

### Restart

#### Adaptive

<p align="center">
<img src="assets/image-20230922181437110.png" alt="image-20230922181437110.png" width="50%">
</p>

## Results

使用库: [Gurobi](https://support.gurobi.com/hc/en-us/articles/360013194392-How-do-I-configure-a-new-Gurobi-C-project-with-Microsoft-Visual-Studio-#:~:text=How%20do%20I%20configure%20a%20new%20Gurobi%20C%2B%2B,the%20Solution%20Explorer%20panel%2C%20then%20select%20Properties.%20), Eigen3, intel MKL(非必要)

|          | PDHG | EGM  | ADMM |
| -------- | ---- | ---- | ---- |
| qap10    |  √    |      |      |
| qap15    |  √    |      |      |
| nug8-3rd | √    |      |      |
| nug20    |      |      |      |

### PDHG

参数：

- $\eta=\frac{0.9}{\sigma_{max}(A)},w=4^2$

| <img width=300px src="./implementation_cpp/fig/PDHG/qap10.png" alt="qap10.png"> | <img width=300px src="./implementation_cpp/fig/PDHG/qap15.png" alt="qap15.png"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img width=300px src="./implementation_cpp/fig/PDHG/nug08-3rd.png" alt="nug08-3rd.png"> |  <img width=300px src="./implementation_cpp/fig/PDHG/nug20.png" alt="nug20.png">  |

### ADMM

<table>
<tr>
<td>
<img src="./implementation_cpp/fig/ADMM/qap10.png" alt="qap10.png"></td>
<td>
<img src="./implementation_cpp/fig/ADMM/qap15.png" alt="qap15.png"></td>
</tr>
<tr>
<td><img src="./implementation_cpp/fig/EGM/nug08-3rd.png" alt="nug08-3rd.png"></td>
<td><img src="./implementation_cpp/fig/EGM/nug20.png" alt="nug20.png"></td>
</table>

## Related links

[Mathematical background for PDLP  | OR-Tools  | Google for Developers](https://developers.google.com/optimization/lp/pdlp_math)

[google-research/FirstOrderLp.jl: Experimental first-order solvers for linear and quadratic programming. (github.com)](https://github.com/google-research/FirstOrderLp.jl)

[google-research/restarting_FOM_for_LP at master · google-research/google-research (github.com)](https://github.com/google-research/google-research/tree/master/restarting_FOM_for_LP)

# carefree-ml

`carefree-ml` implemented Machine Learning algorithms with numpy, mainly for educational use


## Installation

`carefree-ml` requires Python 3.6 or higher.

```bash
git clone https://github.com/carefree0910/carefree-ml.git
cd carefree-ml
pip install -e .
```


## Basic Usage

See `tests/basic_usages.py` for more example

```python
from cfml import *
from cfdata.tabular import TabularDataset

# fetch dataset
boston = TabularDataset.boston()
# make a model
lr = Base.make("linear_regression")
# fit the model
lr.fit(*boston.xy)
# plot loss curve
lr.plot_loss_curve()
# make predictions
predictions = lr.predict(boston.x)
```

...or use methods chaining

```python
import os
from cfml import *
from cfdata.tabular import *

# fetch dataset
prices_file = os.path.join("tests", "datasets", "prices.txt")
prices = TabularData(task_type=TaskTypes.REGRESSION).read(prices_file).to_dataset()
# one liner
Base.make("linear_regression").fit(*prices.xy).visualize1d(*prices.xy).plot_loss_curve()
```


## Supported Algorithms

+ 1-dimensional polynomial fit (`np.polyfit`)
+ Linear Models (Linear Regression, Logistic Regression, Linear SVC, Linear SVR)
+ Naive Bayes (Multinomial NB, Gaussian NB)
+ Support Vector Machine (SVC, SVR)
+ Fully Connected Neural Network (FCNN-clf, FCNN-reg)


## Roadmap

It's up to you! Issues are welcomed :)


## Q & A

+ I used Google Translate to help me translate Chinese to English

### Why carefree-ml?

**为什么选择使用（或借鉴）`carefree-ml`？**

Why shall we choose to use (or learn from) `carefree-ml`?

**`carefree-ml` 其实源于我一直以来未竟的两个心愿**

+ **探索机器学习算法到底可以简化成什么样**
+ **探索各种机器学习算法间的共性究竟有多少**

`carefree-ml` actually stems from my two unfinished wishes

+ Explore how machine learning algorithms can be simplified
+ Explore commonality among various machine learning algorithms

**如果你恰好有这个疑惑，又或是想教导其他人这方面的直观，那么 `carefree-ml` 可能就会比较适合你。但是，如果你对机器学习有着更高的追求，对各种美妙的性质有着探索的欲望，那么 `carefree-ml` 反而可能会激怒你，因为它省略了很多前人研究出来的结晶**

If you happen to have these doubts, or willing to teach others about some intuitions, then `carefree-ml` may be suitable for you. However, if you have a higher pursuit of machine learning and desire to explore more wonderful properties from machine learning, then `carefree-ml` may irritate you, because it omits many of them

**首先，我们知道，机器学习（以及现在大行其道的深度学习）算法，很多时候都可以转化为无约束最优化问题；如果不考虑一些特殊性质（稀疏性，收敛速度等）的话，梯度下降法可以说是一种万金油的方法**

First of all, we know that machine learning (and deep learning) algorithms can often be transformed into unconstrained optimization problems. If some special properties (sparseness, convergence speed, etc.) are not considered, the gradient descent based methods can be the most widely use

**因此，`carefree-ml` 实现的第一大模块，就是一套简单的梯度下降的优化框架，旨在用较少的代码去 handle 大部分情况；在实现一个机器学习算法时，也会优先考虑采用梯度下降算法去实现（这其实就是 `carefree-ml` 所做的最大的简化了，下面的例子就说明了这一点）**

Therefore, the first major module implemented by `carefree-ml` is a simple gradient descent optimization framework, which is designed to handle most cases with less code. After that, when implementing a machine learning algorithm in `carefree-ml`, gradient descent based methods will also be considered first (this is actually the biggest simplification done by `carefree-ml`, the following example of `LinearRegression` illustrates this)

**那么，在这种思想下，我们是如何实现 `LinearRegression` 和 `LogisticRegression` 的呢？首先我们可能都知道：**

+ **两者都是 `线性模型`**
+ **前者做的是 `回归` 问题，后者做的是 `分类` 问题**
+ **后者在输出时使用了 `sigmoid` 激活函数**

So, under this idea, how do we implement `LinearRegression` and` LogisticRegression`? First of all we may all know that:

+ Both of them are `Linear Models`
+ The former deals with `regression` problems, while the latter deals with `classification` problems
+ The latter used `sigmoid` function to output the probability predictions

**但是有一点我们可能之前没注意到：**

+ **如果前者使用 `mse` 作为损失函数，后者使用 `cross_entropy` 作为损失函数，那么它们参数的梯度将会几乎是一模一样的（只差一个倍数）**

But there's one thing that we might not have noticed before:

+ If we use `mse` loss in `LinearRegression` and use `cross_entropy` loss in `LogisticRegression`, then thier parameters' gradients will be almost identical (except for a multiple factor)

**那么，既然它们如此相似，差异点仅在于几个小部分，它们的实现也应该很像才对。所以，在 `carefree-ml` 中，它们的实现的主体将分别是：**

Since they are so similar and the differences are only in a few small parts, their implementation should be very similar as well. Therefore, in `carefree-ml`, the main part of their implementations will be as follows:

```python
class LinearRegression(LinearRegressorMixin, RegressorBase):
    def __init__(self):
        self._w = self._b = None
```

```python
class LogisticRegression(LinearBinaryClassifierMixin, ClassifierBase):
    def __init__(self):
        self._w = self._b = None
        self._sigmoid = Activations("sigmoid")

    def _predict_normalized(self, x_normalized):
        affine = super()._predict_normalized(x_normalized)
        return self._sigmoid(affine)
```

**这里的设计，其实就体现了 `carefree-ml` 想将机器学习算法“简化”的思想。因为我们知道，`mse` loss 下的 `LinearRegression` 是有显式解的（因为就是一个最小二乘法），但是我们仍然用梯度下降去求解它，因为这样它将会与 `LogisticRegression` 共享大部分代码**

The design here actually embodies the idea that `carefree-ml` wants to *simplify* machine learning algorithms. Because we know that `LinearRegression` under` mse` loss has an explicit solution (because it's simply a Least Squares problem), but we still use gradient descent to solve it because in this case it will share most of its code with `LogisticRegression`'s code

**当然，这种简化（将许多算法都归结为无约束优化问题并用梯度下降法求解）并不是全是坏处，比如我们完全可以在代码几乎不变的前提下，去求解 `l1` loss、或者其它形形色色的 loss 下的 `LinearRegression`**

Of course, this simplification (reducing many algorithms to unconstrained optimization problems and solving them by gradient descent) has its advantage too. For example, we can solve `l1` loss or other losses in `LinearRegression` under the premise that the corresponding training codes will be almost unchanged

**再比如说 `svm`。虽然支持向量分类和支持向量回归看上去是非常不一样的两种算法，但是抽丝剥茧之后，如果用梯度下降法去求解，就会发现其实大部分代码仍然是共享的，这也恰好辅证了为何它们同属 `svm` 的范畴：**

Another example is `svm`. Although support vector classification and support vector regression seem to be very different algorithms, but after pulling the cocoon, if you use gradient descent based methods to solve them, you will find that most of the codes are still shared. This also justifies why they belong to the same category - `svm`:

```python
class CoreSVCMixin:
    @staticmethod
    def _preprocess_data(x, y):
        y_svm = y.copy()
        y_svm[y_svm == 0] = -1
        return x, y_svm

    @staticmethod
    def get_diffs(y_batch, predictions):
        return {"diff": 1. - y_batch * predictions, "delta_coeff": -y_batch}

class SVCMixin(BinaryClassifierMixin, SVMMixin, metaclass=ABCMeta):
    def predict_prob(self, x):
        affine = self.predict_raw(x)
        sigmoid = Activations.sigmoid(np.clip(affine, -2., 2.) * 5.)
        return np.hstack([1. - sigmoid, sigmoid])
```

```python
class CoreSVRMixin:
    def get_diffs(self, y_batch, predictions):
        raw_diff = predictions - y_batch
        l1_diff = np.abs(raw_diff)
        if self.eps <= 0.:
            tube_diff = l1_diff
        else:
            tube_diff = l1_diff - self.eps
        return {"diff": tube_diff, "delta_coeff": np.sign(raw_diff)}

class SVRMixin(SVMMixin, metaclass=ABCMeta):
    def predict(self, x):
        return self.predict_raw(x)
```

**然后真正实现 `svm` 算法时，就只需继承不同的类即可：**

After these, when you actually implement the `svm` algorithms, you only need to inherit different classes:

```python
class SVC(CoreSVCMixin, SVCMixin, ClassifierBase):
    def __init__(self,
                 kernel: str = "rbf"):
        self._kernel = Kernel(kernel)
```

```python
class SVR(CoreSVRMixin, SVRMixin, RegressorBase):
    def __init__(self,
                 eps: float = 0.,
                 kernel: str = "rbf"):
        self._eps = eps
        self._kernel = Kernel(kernel)
```

> **当然了，真正的核心代码（`SVMMixin`）还是要写一写的**
>
> Of course, the real core codes (`SVMMixin`) still have to be written

**此外，除了相似算法间的代码共享，`carefree-ml` 还致力于常见工程功能上的代码共享。比如说，我们一般可能需要：**

+ **对输入的特征进行规范化处理（normalization）**
+ **在回归问题中对标签进行 normalization**
+ **在二分类问题中通过 roc curve 以及具体的 metric 来挑选出最优分类阈值**

In addition, besides code sharing between similar algorithms, `carefree-ml` is also dedicated to sharing codes on common engineering functions. For example, we may generally need:

+ Normalize the input features
+ Normalize the labels in regression problems
+ Utilize roc curve to find the best threshold of specific metric in binary classification problems

**这些工程上的东西，也是理应进行代码共享的。因此，`carefree-ml` 确实在 `cfml.models.mixins` 中，实现了  `NormalizeMixin` 和 `BinaryClassifierMixin`，用于实现这些可能被广泛运用的功能**

These engineering functions are also supposed to share codes. Therefore, `carefree-ml` implements ` NormalizeMixin` and `BinaryClassifierMixin` in `cfml.models.mixins` for these functions that may be widely used


### What can carefree-ml do?

**`carefree-ml` 能做到什么？**

What can `carefree-ml` do? 

**首先，最近其实有很多用 `numpy` 实现海量算法的 repo，所以单单用 `numpy` 来作为卖点是不合适的。我个人认为的话，`carefree-ml` 之所以还算有些特色，主要是由于如下三点：**

+ **实现了一个轻量级的、泛用性比较好的梯度下降框架**
+ **比起模型的性能，更注重于让算法间共享逻辑、代码；正因此，总代码量会比较少**
+ **即使在第二点的“桎梏”下，在小的、比较简单的数据集上，无论是速度还是性能，都是可以锤掉 `scikit-learn` 的友商产品的**

First of all, there are actually a lot of repos that use `numpy` to implement massive algorithms recently, so it is not appropriate to use` numpy` as a selling point. In my personal opinion, the reason why `carefree-ml` is still special are shown as follows:

+ Implemented a lightweight gradient descent framework which can be used in a wide range of problems
+ Compared with the performance of the model, it focused more on the sharing of logic and codes between algorithms. Therefore, the total amount of code will be less
+ Even under the 'shackles' of the second point, on small and relatively simple datasets, it can beat `scikit-learn` in either speed or performance to some extend

**测试方式（包括了安装步骤）：**

Here's how you can test it (included installation procedures)

```bash
git clone https://github.com/carefree0910/carefree-ml.git
cd carefree-ml
pip install -e .
cd tests/unittests
python test_all.py
```

**在输出中随便截几组数据吧：**

Here's some fragments from the outputs:

```text
~~~  [ info ] timing for    cfml_fcnn     : 0.310764
~~~  [ info ] timing for   sklearn_fcnn   : 0.549960
==========================================================
|             cfml_fcnn  |    mae     |  2.682794  |  <-  
|          sklearn_fcnn  |    mae     |  3.969561  |
----------------------------------------------------------
===========================================================
|             cfml_fcnn  |    mse     |  15.635315  |  <-  
|          sklearn_fcnn  |    mse     |  30.890426  |
-----------------------------------------------------------
```

```text
~~~  [ info ] timing for     cfml_lr      : 0.039881
~~~  [ info ] timing for    sklearn_lr    : 0.654799
==========================================================
|               cfml_lr  |    auc     |  0.996287  |  <-  
|            sklearn_lr  |    auc     |  0.994675  |
----------------------------------------------------------
==========================================================
|               cfml_lr  |    acc     |  0.980668  |  <-  
|            sklearn_lr  |    acc     |  0.957821  |
----------------------------------------------------------
```

```text
# gaussian naive bayes
~~~  [ info ] timing for     cfml_gnb     : 0.000000
~~~  [ info ] timing for   sklearn_gnb    : 0.001028
# multinomial naive bayes
~~~  [ info ] timing for     cfml_mnb     : 0.003990
~~~  [ info ] timing for   sklearn_mnb    : 0.007011
```

```text
~~~  [ info ] timing for     cfml_svc     : 0.207024
~~~  [ info ] timing for    cfml_l_svc    : 0.023937
~~~  [ info ] timing for    sklearn_lr    : 0.571722
~~~  [ info ] timing for   sklearn_svc    : 0.007978
~~~  [ info ] timing for  sklearn_l_svc   : 0.148603
==========================================================
|            cfml_l_svc  |    auc     |  0.996300  |
|              cfml_svc  |    auc     |  1.000000  |  <-  
|            sklearn_lr  |    auc     |  0.994675  |
----------------------------------------------------------
==========================================================
|            cfml_l_svc  |    acc     |  0.985940  |
|              cfml_svc  |    acc     |  1.000000  |  <-  
|         sklearn_l_svc  |    acc     |  0.848858  |
|            sklearn_lr  |    acc     |  0.957821  |
|           sklearn_svc  |    acc     |  0.922671  |
----------------------------------------------------------
```

```text
~~~  [ info ] timing for     cfml_svr     : 0.090758
~~~  [ info ] timing for    cfml_l_svr    : 0.027925
~~~  [ info ] timing for   sklearn_svr    : 0.008012
~~~  [ info ] timing for  sklearn_l_svr   : 0.165730
==========================================================
|            cfml_l_svr  |    mae     |  3.107422  |  <-  
|              cfml_svr  |    mae     |  5.106989  |
|         sklearn_l_svr  |    mae     |  4.654314  |
|           sklearn_svr  |    mae     |  5.259882  |
----------------------------------------------------------
===========================================================
|            cfml_l_svr  |    mse     |  24.503884  |  <-  
|              cfml_svr  |    mse     |  66.583145  |
|         sklearn_l_svr  |    mse     |  39.598211  |
|           sklearn_svr  |    mse     |  66.818898  |
-----------------------------------------------------------
```

**当然了，吹是这么吹，最后我们还是得负责任地说一句：从实用性、泛化性来说，`scikit-learn` 肯定是吊打 `carefree-ml` 的（比如 `carefree-ml` 完全不支持稀疏数据）。但是，正如我一开始所说，`carefree-ml` 只是想探索“机器学习算法到底可以简化成什么样”的产物，所以在简单的数据集上，拟合能力、拟合速度超过 `scikit-learn` 其实也并不奇怪**

Of course, in the end we still have to say something responsibly: from the perspective of practical use and generalization, `scikit-learn` will beat ` carefree-ml` by all means (for short, `carefree-ml` does not support sparse data). However, as I said at the beginning, `carefree-ml` focus on exploring *how machine learning algorithms can be simplified*. So it is not surprising that `carefree-ml` can exceed `scikit-learn` on small & simple datasets in fitting capacity & fitting speed.

> **注：上述实验结果都是训练集上的结果，所以只能反映拟合能力，不能反映泛化能力**
>
> Notice that the above experimental results are the results on the training set, so it can only reflect the fitting capacity, not the generalization capacity


### How can I utilize carefree-ml?

**我能怎么使用 `carefree-ml`？**

How can I utilize `carefree-ml`?

**从实用性角度来看，也许 `carefree-ml` 实现的那套简易梯度下降框架，是相对而言最实用的。但即便是它，也会被 `pytorch` 轻松吊锤**

From a practical point of view, perhaps the lightweight gradient descent framework implemented by `carefree-ml` is relatively the most useful tool. But even it will be easily defeated and replaced by `pytorch`

**所以，正如我开头所说，mainly for educational use，可能教育意义会大于实用意义。虽然本人学术能力不咋地，但是毕竟该 repo 的初衷应该很少搞学术的会看得起并加以研究，所以从这个角度来看，`carefree-ml` 也许能给你带来一些新的体会**

So, as I said at the beginning, `carefree-ml` is mainly for educational use, so the meaning of education may be greater than the practical meaning. Although my academic ability is not good at all, the original intention of this repo might not be worthy of academic researching. So from this perspective, `carefree-ml` may give you some new sights


## License

`carefree-ml` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-ml/blob/master/LICENSE) file.

---

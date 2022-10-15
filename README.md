[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)

# Pgx

Highly parallel game simulator for reinforcement learning.

## Basic API
Pgx's basic API consists of *pure functions* following the JAX's design principle.

```py

@dataclass
class State:
  i: np.ndarray = jnp.zeros(1)
  board: np.ndarray = jnp.array((10, 10))


@jax.jit
def init(rng: jnp.ndarray, **kwargs) -> State:
  return State()

@jax.jit
def step(state: State, action: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
  return State(), r, terminated

@jax.jit
def observe(state: State) -> jnp.ndarray:
  return jnp.ones(...)

```

## Roadmap

|Game|Logic| Jit                                                                                                                      |Baseline|Visualization|Gym/PettingZoo|
|:---|:---|:-------------------------------------------------------------------------------------------------------------------------|:---|:---|:---|
|TicTacToe||||||
|AnimalShogi| :white_check_mark: | :white_check_mark:                                                                                                       ||||
|MiniMahjong| :white_check_mark: | :white_check_mark:                                                                                                       ||||
|MinAtar <br>[kenjyoung/MinAtar](https://github.com/kenjyoung/MinAtar)|-| :white_check_mark: Asterix<br> :white_check_mark: Breakdown<br> :white_check_mark: Freeway<br> :white_check_mark: Seaquest<br> :white_check_mark: SpaceInvaders ||||
|Chess||||||
|Shogi| :construction: |||||
|Go| :white_check_mark: | :white_check_mark:                                                                                                       ||||
|ContractBridgeBidding||||||
|Backgammon||||||
|Mahjong| :construction: |||||

## Development guide (in Japanese)

### Jax概要

#### なぜJaxを使うのか

Jaxの `jax.numpy` はNumpyと同じAPIで利用できる自動微分ライブラリです。
PyTorchと比べて、Numpyを知っていれば新しくAPIを学習する必要がないというメリットがあります。
その他にこのプロジェクトで利用する重要な機能として、`jax.jit` と `jax.vmap` があります。

`jax.jit` は関数をJIT (Just In Time Compilation) によって実行しているアクセラレータ（CPU/GPU/TPU）に特化したコードにコンパイルしてから実行することができます。
これによってGPU/TPUを利用した効率的な演算が可能になります。
次の例では、最初の実行時にforの結果をそのまま返すコードにコンパイルされています。

```py
>>> def f():
...    s = 0
...    for i in range(1, 11):
...        s += i
...    return s

>>> f_jit = jax.jit(f)
>>> f_jit()
55
>>> jax.make_jaxpr(f)()
{ lambda ; . let  in (55,) }
```

`jax.vmap` を利用すると、自動で関数をバッチ化することができます。
最初の次元が自動的にバッチサイズに対応します。
これによって、並列化を全く意識せずにコードを書いても、あとから簡単にGPU/TPU上で並列化することができます。

```py
>>> def f(n):
...     return jax.numpy.ones(3) * n

>>> f_vmap = jax.vmap(f)
>>> f_vmap(jnp.arange(3))
[[0. 0. 0.]
 [1. 1. 1.]
 [2. 2. 2.]]
```

Pgxでは、Jaxを使ってゲームのシミュレータを書くことで、GPU/TPU上で高速かつ並列化可能なゲームの実装を目指します。
似たような目的で、同様にJaxを使って高速なシミュレータを実装しているものとして[Brax](https://github.com/google/brax)があります。

### Jitの制約

`jax.jit` を使うことで、アクセラレータに特化したコードへの変換が可能だと説明しましたが、
**任意のPythonコードがJIT可能なわけではありません。**
GPU/TPU上での効率的な演算を可能にするため、例えば `ndarray` はstaticである必要があります（実行前にサイズが決まっている必要があります）。
Jit化できないコードの例として次のようなものがあります。

<table>
<tr>
<td> Jit不可な事例 </td> <td> コード </td>
</tr>
<tr>
<td> 動的な配列 </td>
<td>

```py
@jax.jit
def f(N):
  return np.ones(N)
```

</td>
</tr>

<tr>
<td> 引数を条件にしたif </td>
<td>

```py
@jax.jit
def f(n):
  if n == 0:
    return 0
  else:
    return 1
  
```

</td>
</tr>

<tr>
<td> 引数で回数が決まるfor </td>
<td>

```py
@jax.jit
def f(n):
  s = 0
  for i in range(n):
    s += i
  return s
```

</td>
</tr>

<tr>
<td> 引数を条件にしたwhile </td>
<td>

```py
@jax.jit
def f(n):
  s = 0
  while i < n:
    s += i
    i += 1
  return s
```

</td>
</tr>


</table>

これらのコードに共通するのは、**実行時に必要な情報が決まっていないということです。**
例えば行列のサイズやforの回数が決まっていません。
またifでは条件分岐したときに返り値の値が同じ型なのかどうかの保証がありません。

### `jax.jit` 可能なコードへの変換例

上述のように、Jit可能なコードには制限がありますが、 `jax.lax` を使うことで、これらの制約を緩和してプログラムを書くことができます。
`jax.lax` は関数型プログラミングでのコーディングを強制することでコンパイラがコードを変換するのに必要な情報を保証します。
ここでは、if/for/whileについて、どのようにコードを書き換えたら良いのかを列挙します。


<table>
<tr>
<td> 

ドキュメント（等価なコード）

</td> 
<td> 

例 Before

</td>
<td> 例 After </td>
</tr>


<tr>
<td> 

if: [`jax.lax.cond`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html)

```py
def cond(pred, true_fun, false_fun, 
         *operands):
  if pred:
    return true_fun(*operands)
  else:
    return false_fun(*operands)
```

Note: `true_fn` と `false_fn` の返り値の型が同じで必要があります。

</td>
<td>

```py
def f(n):
  if n == 0:
    return jnp.zeros(3) 
  else:
    return jnp.ones(3)
```

</td>
<td>

```py
@jax.jit
def f(n):
  return jax.lax.cond(
    n == 0:
    lambda: jnp.zeros(3),
    lambda: jnp.ones(3)
  )
```



</td>
</tr>


<tr>
<td> 

for: [`jax.lax.fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html)

```py
def fori_loop(lower, upper, 
              body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val
```

Note: 一つの変数valに繰り返し同じ操作を行う場合、 `fori_loop` を使う。

</td>
<td>

```py
def f(n):
  s = 0
  for i in range(1, n + 1):
    s += i
  return s
```

</td>
<td>

```py
@jax.jit
def f(n):
  s = jax.lax.fori_loop(
    1, n + 1,
    lambda i, x: x + i
  )
  return s
# Practically, use jnp.sum()
```



</td>
</tr>


<tr>
<td> 

for: [`jax.lax.map`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.map.html)

```py
def map(f, xs):
  return np.stack([f(x) for x in xs])
```

Note: いわゆる関数型プログラミングのmap操作を行う。配列の各要素に対して作用させたい場合に使う。

</td>
<td>

```py
def f(n):
  arr = jnp.arange(10)
  l = [is_odd(arr[i]) for i in range(10)]
  return jnp.array(l)
```

</td>
<td>

```py
@jax.jit
def f(n):
  arr = jnp.arange(10)
  return jax.lax.map(
    lambda: is_odd(arr[i]), 
    jnp.arange(10)
  )
# jax.lax.map(is_odd, arr) is enough in this case
# Access to out of scope array is often used
```



</td>
</tr>




<tr>
<td> 

for: [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)

```py
def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
```

Note: `fori_loop` と `map` の操作を同時に行うことができる。

</td>
<td>

```py
# apply abs and find argmax index
def f(n):
  ix = -1
  m = 0
  arr = jnp.array([5, -7, 3])
  abs_arr = jnp.zeros_like(arr)
  for i in range(3):
    abs_arr[i] = abs(arr[i])
    if m < abs_arr[i]:
      ix = i
  return ix, abs_arr
```

</td>
<td>

```py
@jax.jit
def f(n):
  m = 0
  arr = jnp.array([5, -7, 3])

  def each_fn(ix, i):
    val = abs(arr[i])
    if m < val:
      ix = i
    return i, val
      
  return jax.lax.scan(
    each_fn, -1, jnp.arange(3)
  )
```



</td>
</tr>





<tr>
<td> 

while: [`jax.lax.while_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html)

```py
def while_loop(cond_fun, 
         body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val
```

Note: 

</td>
<td>

```py
def f(n):
  s = 0
  i = 1
  while s + (i * i) < n:
    s += i * i
    i += 1
  return s
```

</td>
<td>

```py
@jax.jit
def f(n):
  def cond_fn(x):
    i, s = x
    return s + (i * i) < n

  def body_fn(x):
    i, s = x
    return i + 1, s + (i * i)

  return jax.lax.while_loop(
    cond_fn,
    body_fn,
    (1, 0)
  )
```

</td>
</tr>

</table>






### Tips

* いきなり `jax.lax` を使って実装するのは難しいので、まずNumpyでロジックとテストを書き、テストが通るようにNumpyをjax.numpyへ書き換え、そのあと少しずつJit可能なコードへ書き換えるという段階を経るのが良い。
* なるべくfor/whileではなくNumpyでの行列演算ができないか考える。
* 書き換えのテストしやすさを考慮し、実装は細かい純粋関数に分ける。細かすぎて困ることはない。Numpy実装で各十行以内くらいが一つの目安。
* forを回すときは、carryの更新か、map操作のどちらかにする（あるいはそれらの組み合わせ）。
* break/continueはなるべく避ける。どうしても必要な場合には `jax.lax.while` の使用を考える。
* 早期リターンが必要な場合には、早期リターン以後のロジックを別関数に切り分けて
* 可変長listは絶対に使わない（append/deleteは使わない）

### 開発手順


## LICENSE

TDOO

* MinAtar is GPL-3.0 License

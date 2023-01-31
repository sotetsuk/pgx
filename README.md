[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)

# Pgx

A collection of GPU/TPU-accelerated game simulators for reinforcement learning.

## APIs
Pgx's basic API consists of *pure functions* following the JAX's design principle.
This is to explicitly let users know that state transition is determined ONLY from `state` and `action` and to make it easy to use `jax.jit`.
Pgx defines the games as AEC games (see PettingZoo paper), in which only one agent acts and then turn changes.


### Design goal
1. Be explicit
2. Be simple than be universal


### Usage

```py
import jax
import pgx

num_batch = 100

init, step, observe, info = pgx.make(env_id="Go-5x5",)
init = jax.jit(jax.vmap(init))
step = jax.jit(jax.vmap(step))
observe = jax.jit(jax.vmap(observe))

models = {0: ..., 1: ...}

rng = jax.random.PRGNKey(999)
keys = jax.random.split(rng, num_batch)

state = init(keys)
total_reward = jnp.zeros(batch_size, dtype=jnp.float32)
while not (state.terminated).all():
    observations = [observe(state, player_id) for player_id in (0, 1)]
    action = jnp.where(
        state.curr_player == 0,
        models[0](observations[0]),
        models[1](observations[1]),
    )
    state = step(obs, action)
    total_reward += reward
```

### API Description

```py
# N: num agents
# A: action space size
# M: observation dim
@dataclass
class State:
    rng: jax.random.KeyArray  # necessary for autoreset
    curr_player: jnp.ndarray
    # 0 ~ N-1. Different from turn (e.g., white/black in Chess) 
    # Behavior is undefined when terminated (set -1 is inconvenient in batch situation)
    reward: jnp.ndarray
    terminated: jnp.ndarray
    legal_action_mask: jnp.ndarray
  

def init(rng: jnp.ndarray) -> State:
  return state 

# step is deterministic by default
# if state.terminated is True, state.reward is set to zero and the other fields are unchanged
def step(state: State, 
         action: jnp.ndarray)
    -> State:
  return state  # rewards: (N,) 

def observe(state: State, 
            player_id: jnp.ndarray) 
    -> jnp.ndarray:
  # Zero array if state.curr_player is -1
  return obs 

# replace state.rng or shuffle hidden states (e.g., unopened public cards)
def shuffle(state: State, rng: Optional[jnp.ndarray]) 
    -> State:
   return state
```

### Limitations (for the simplicity)
* Does **NOT** support agent death and creation, which dynmically changes the array size. It does not well suit to GPU-accelerated computation.
* Does **NOT** support Chance player (Nature player) with action selection.
* Does **NOT** support OpenAI Gym API.
    * OpenAI Gym is for single-agent environment. Most of Pgx environments are multi-player games. Just defining opponents is not enough for converting multi-agent environemnts to OpenAI Gym environment. E.g., in the game of go, the next state s' is defined as the state just after placing a stone in AlhaGo paper. However, s' becomes the state after the opponents' play. This changes the definition of V(s').
* Does **NOT** support PettingZoo API.
    * PettingZoo is *Gym for multi-agent RL*. As far as we know, PettingZoo does not support vectorized environments (like VectorEnv in OpenAI Gym). As Pgx's main feature is highly vectorized environment via GPU/TPU support, We do not currently support PettingZoo API. 



### `skip_chance`
* We prepare skip_chance=True option for some environments. This makes it possible to consider value function for "post-decision states" (See AlgoRL book). However, we do not allow chance agent to choose action like OpenSpiel. This is because the action space of chance agent and usual agent are different. Thus, when the chance player is chosen (`current_player=-1`), `action=-1` must be returned to step function. Use `shuffle` to make `step` stochastic.

### truncatation and auto_reset
* supported by `make(env_id="...", auto_reset=True, max_episode_length=64)`
* `auto_reset` will replace the terminal state by initial state (but `is_terminal=True` is set)
* `is_truncated=True` is also set to state

### Concerns
* For efficient computation, current_player must be synchronized? but it seems difficult (or impossible?). It is impossible to synchronize the terminations.

## Roadmap

|Game|Logic| Jit                                                                                                                      |Visualization|Speed benchmark|Baseline|
|:---|:---|:-------------------------------------------------------------------------------------------------------------------------|:---|:---|:---|
| Tic-tac-toe | :white_check_mark: | :white_check_mark: ||||
| [Animal Shogi](https://en.wikipedia.org/wiki/D%C5%8Dbutsu_sh%C5%8Dgi) | :white_check_mark: | :white_check_mark:                                                                                                       | :white_check_mark: |||
| [Sparrow Mahjong](https://sugorokuya.jp/p/suzume-jong) |  |                                                                                                        ||||
| [MinAtar](https://github.com/kenjyoung/MinAtar)|-| :white_check_mark: Asterix<br> :white_check_mark: Breakdown<br> :white_check_mark: Freeway<br> :white_check_mark: Seaquest<br> :white_check_mark: SpaceInvaders ||||
|Chess| :white_check_mark: ||:construction:|||
|Shogi| :white_check_mark: || :white_check_mark: |||
|Go| :white_check_mark: | :white_check_mark:                                                                                                       |:white_check_mark: |||
|Backgammon| :construction: ||:construction:|||
|Bridge Bidding| :construction: |||||
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

### Numpyとの違い

`jax.numpy` はNumpyと同じAPIを持つと説明しましたが、違いもあります。
それは `jax.numpy` の `ndarray` は **immutable** だということです。
そのため、in-placeな更新はできません。

```py
>>> a = jax.numpy.zeros(3)
>>> a[1] = 1  # Error!
```

代わりに、 `.at[i].set(val)` を使って新しいオブジェクトを作る必要が有ります。


```
>>> a = a.at[1].set(1)
```

スライスも利用できます。また、`set` の他にも `add` といったメソッドが利用できます。


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

Note: `true_fn` と `false_fn` の返り値の型が同じで必要があります。`and/or` の代わりに `&/|` を使います。

</td>
<td>

```py
def f(n):
  if (n % 2 == 0 and 
      n % 3 == 0):
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
    (n % 2 == 0) & (n % 3 == 0):
    lambda: jnp.zeros(3),
    lambda: jnp.ones(3)
  )
```



</td>
</tr>



<tr>
<td> 

if: [`jax.lax.switch`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html)

```py
def switch(index, branches, 
           *operands):
  index = clamp(0, index, 
            len(branches) - 1)
  return branches[index](
    *operands
  )
```

Note: 3つ以上の条件分岐のときなどに使えます。返り値の型は同じである必要があります。

</td>
<td>

```py
def f(n):
  if n == 0:
    return jnp.zeros(3) 
  elif n == 1:
    return jnp.ones(3) 
  elif n == 2:
    return jnp.ones(3) * 2
```

</td>
<td>

```py
@jax.jit
def f(n):
  return jax.lax.switch(
    n,
    [lambda: jnp.zeros(3),
     lambda: jnp.ones(3),
     lambda: jnp.ones(3) * 2],
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
  return np.stack(
    [f(x) for x in xs]
  )
```

Note: いわゆる関数型プログラミングのmap操作を行う。配列の各要素に対して作用させたい場合に使う。

</td>
<td>

```py
def f(n):
  arr = jnp.arange(10)
  l = [is_odd(arr[i]) 
        for i in range(10)]
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
def scan(f, init, xs, 
         length=None):
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
# apply abs and find argmax
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

Note: `break` を使いたいときなどに使用を考える。

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
* 書き換えのしやすさを考慮し、ifやforの深いネストは避け、細かい純粋関数に分ける。細かすぎて困ることはない。Numpy実装で各十行以内くらいが一つの目安。
* forを回すときは、carryの更新か、map操作のどちらかになるように実装する（あるいはそれらの組み合わせ）。
* 通常のforはかなり遅いので、固定長であってもすべて `fori_loop` 等で書き換える。
* Numpy実装の時点でbreak/continueはなるべく避ける。
* 早期リターンが必要な場合には、早期リターン以後のロジックを別関数に切り分ける。
* 可変長listは絶対に使わない（append/deleteは使わない）

### 開発手順

* (1) Numpyを使って実装する
  * まずは深く考えずにNumpyでロジックを実装します。
    * 深いネストはなるべく避けます。
    * 細かい純粋関数へ分割することを意識します。
    * forはなるべく避け、使うときはcarryの更新かmap操作となるようにします。
  * このとき、テストをちゃんと書くのが重要です。後の書き換えが楽になります。
  * `State` クラスを `dataclass` を使って実装します。 `State` クラスの全ての要素は `ndarray` にして、型も指定します（要素が一つでも）。
    * `State` クラスのデータの持ち方は多少冗長にしたほうが実装が楽になる場合が多いです。
  * `init`, `step`, `observe` を実装します。
  * 実装がJit化可能かどうか検証・修正します。
* (2) `jax.numpy` に書き換える
  * `import numpy as np` を `import jax.numpy as jnp` とします。
  * `State` クラスを `flax.struct.dataclass` にします。
  * in-placeな配列操作を `at` を使って書き換えます。
* (3) `jax.jit` 可能な形に書き換える
  * テストが通ることを確認しつつ `@jax.jit` を `def` の上に一つずつ付けていく。
  * 関数呼び出しの一番末尾のものからJit化していきます（`step`が一番最後になる）。


## LICENSE

TDOO

* MinAtar is GPL-3.0 License

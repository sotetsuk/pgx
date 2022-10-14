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

`jax.numpy` は `numpy` と同じAPIで利用できる自動微分ライブラリです。
PyTorchと比べて、Numpyを知っていれば新しくAPIを学習する必要がないというメリットがあります。
その他にこのプロジェクトで利用する重要な機能として、`jax.jit` と `jax.vmap` があります。

**jax.jit** は関数をJIT (Just In Time Compilation) によってアクセラレータ（GPU/TPU）に特化したコードに書き換えることができます。
これによって通常のPythonコードであっても、GPU/TPUを利用した効率的な演算が可能になります。

**jax.vmap** を利用すると、自動で関数をバッチ化することができます。
最初の次元が自動的にバッチサイズに対応します。
これによって、並列化を全く意識せずにコードを書いても、あとから簡単にGPU/TPU上で並列化することができます。

Pgxでは、Jaxを使ってゲームのシミュレータを書くことで、GPU/TPU上で高速かつ並列化可能なゲームの実装を目指します。

#### Jitの制約

`jax.jit` を使うことで、アクセラレータに特化したコードへの変換が可能だと説明しましたが、
**任意のPythonコードが変換可能なわけではありません。**
GPU/TPU上での効率的な演算を可能にするため、


TODO: Jit不可能な例（if, for, 早期リターン）

#### Tips

* いきなり `jax.lax` を使って実装するのは難しいので、まずNumpyでロジックとテストを書き、テストが通るようにNumpyをjax.numpyへ書き換え、そのあと少しずつJit可能なコードへ書き換えるという段階を経るのが良い。
* なるべくfor/whileではなくNumpyでの行列演算ができないか考える。
* 書き換えのテストしやすさを考慮し、実装は細かい純粋関数に分ける。細かすぎて困ることはない。Numpy実装で各十行以内くらいが一つの目安。
* forを回すときは、carryの更新か、map操作のどちらかにする（あるいはそれらの組み合わせ）。
* break/continueはなるべく避ける。どうしても必要な場合には `jax.lax.while` の使用を考える。
* 早期リターンが必要な場合には、早期リターン以後のロジックを別関数に切り分けて
* 可変長listは絶対に使わない（append/deleteは使わない）

#### `jax.jit` 可能なコードへの変換例


### 開発手順


## LICENSE

TDOO

* MinAtar is GPL-3.0 License

"""展柜环境越界剂量分析。

对相邻读数做**线性插值**，精确求出温/湿度与阈值的交点；越限幅度
（高于上限或低于下限的差值）对时间积分得到“剂量”：

    温度剂量 = ∫ max(0, T - temp_max) + max(0, temp_min - T) dt   （°C·秒）
    湿度剂量 = ∫ max(0, H - hum_max)  + max(0, hum_min - H) dt   （%RH·秒）

规则：

1. 相邻读数时间间隔**严格超过** ``max_gap_seconds`` 时不在两点间插值，
   时间线在此断开，断开的未知区间不计剂量、也不合并事件
   （间隔恰好等于 ``max_gap_seconds`` 仍视为连续）。
2. 交点按线性函数精确求解（有理数比例），不以采样点近似。
3. 温度、湿度各自的越界连续区间先独立生成，再把**任一维度**越界的
   相交或首尾相接区间合并为一个事件。事件按开始时间稳定排序，
   相同开始时间时温度区间优先。

全程纯函数、确定性，不依赖任何外部服务。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .segmentation import Sample

TEMPERATURE = "temperature"
HUMIDITY = "humidity"
_DIM_ORDER = (TEMPERATURE, HUMIDITY)
_DIM_RANK = {name: i for i, name in enumerate(_DIM_ORDER)}

HIGH = "high"
LOW = "low"


@dataclass(frozen=True)
class Thresholds:
    """单一维度的可选上下限；两者皆为 None 时该维度不参与判定。"""

    low: float | None = None
    high: float | None = None

    def enabled(self) -> bool:
        return self.low is not None or self.high is not None


@dataclass(frozen=True)
class _Piece:
    """一个维度在一段连续时间上的越界子区间（内部不可再被阈值交点切开）。"""

    dim: str
    start: dt.datetime
    end: dt.datetime
    dose: float
    excess: float  # 该子区间端点上的最大越限幅度（线性函数极值必在端点）
    value: float  # 取得 excess 时的物理读数（方向上的极值）
    direction: str  # high：越上限；low：越下限


@dataclass
class ExcursionEvent:
    index: int
    start_ts: dt.datetime
    end_ts: dt.datetime
    duration_seconds: float
    dimensions: list[str]
    peak_temperature: float | None
    peak_humidity: float | None
    temperature_peak_excess: float
    humidity_peak_excess: float
    temperature_dose: float
    humidity_dose: float


@dataclass
class _Accumulator:
    """事件合并过程中的累计状态。"""

    start: dt.datetime
    end: dt.datetime
    dims: set[str]
    doses: dict[str, float]
    # 每个维度两个方向上的 (最大越限幅度, 对应读数)
    best: dict[str, dict[str, tuple[float, float]]]

    @classmethod
    def from_piece(cls, piece: _Piece) -> "_Accumulator":
        acc = cls(
            start=piece.start,
            end=piece.end,
            dims={piece.dim},
            doses={d: 0.0 for d in _DIM_ORDER},
            best={d: {} for d in _DIM_ORDER},
        )
        acc._absorb(piece)
        return acc

    def _absorb(self, piece: _Piece) -> None:
        self.doses[piece.dim] += piece.dose
        cur = self.best[piece.dim].get(piece.direction)
        if cur is None or piece.excess > cur[0] or (
            piece.excess == cur[0]
            and (piece.value > cur[1] if piece.direction == HIGH else piece.value < cur[1])
        ):
            self.best[piece.dim][piece.direction] = (piece.excess, piece.value)

    def merge(self, piece: _Piece) -> None:
        if piece.end > self.end:
            self.end = piece.end
        self.dims.add(piece.dim)
        self._absorb(piece)

    def _peak(self, dim: str) -> tuple[float | None, float]:
        cand = self.best[dim]
        if not cand:
            return None, 0.0
        # 越限幅度更大的方向优先；幅度相同取上限方向，保证确定性
        direction = HIGH if cand.get(HIGH, (-1.0,))[0] >= cand.get(LOW, (-1.0,))[0] else LOW
        excess, value = cand[direction]
        return value, excess

    def finalize(self, index: int) -> ExcursionEvent:
        t_peak, t_excess = self._peak(TEMPERATURE)
        h_peak, h_excess = self._peak(HUMIDITY)
        return ExcursionEvent(
            index=index,
            start_ts=self.start,
            end_ts=self.end,
            duration_seconds=max(0.0, (self.end - self.start).total_seconds()),
            dimensions=[d for d in _DIM_ORDER if d in self.dims],
            peak_temperature=t_peak,
            peak_humidity=h_peak,
            temperature_peak_excess=max(0.0, t_excess),
            humidity_peak_excess=max(0.0, h_excess),
            temperature_dose=max(0.0, self.doses[TEMPERATURE]),
            humidity_dose=max(0.0, self.doses[HUMIDITY]),
        )


def _dimension_pieces(
    a: Sample,
    b: Sample,
    duration: float,
    dim: str,
    thr: Thresholds,
) -> list[_Piece]:
    """求相邻读数 ``a -> b`` 之间某一维度的全部越界子区间。

    温度/湿度在区间内是时间的线性函数；把与每个已配置阈值的交点作为
    切分点，子区间内相对任一阈值的符号恒定，越限幅度也是线性的，
    梯形积分即精确解。
    """
    v0 = getattr(a, dim)
    v1 = getattr(b, dim)

    cuts = {0.0, duration}
    for bound in (thr.high, thr.low):
        if bound is None:
            continue
        d0, d1 = v0 - bound, v1 - bound
        if d0 * d1 < 0.0:  # 严格异号：交点严格落在区间内部
            cuts.add(duration * d0 / (d0 - d1))

    xs = sorted(cuts)
    pieces: list[_Piece] = []
    for x0, x1 in zip(xs, xs[1:]):
        width = x1 - x0
        if width <= 0.0:
            continue
        va = v0 + (v1 - v0) * (x0 / duration)
        vb = v0 + (v1 - v0) * (x1 / duration)
        vm = v0 + (v1 - v0) * (((x0 + x1) / 2.0) / duration)

        if thr.high is not None and vm > thr.high:
            e0 = max(0.0, va - thr.high)
            e1 = max(0.0, vb - thr.high)
            excess = max(e0, e1)
            value = vb if e1 >= e0 else va
            pieces.append(
                _Piece(
                    dim=dim,
                    start=a.ts + dt.timedelta(seconds=x0),
                    end=a.ts + dt.timedelta(seconds=x1),
                    dose=(e0 + e1) * 0.5 * width,
                    excess=excess,
                    value=value,
                    direction=HIGH,
                )
            )
        elif thr.low is not None and vm < thr.low:
            e0 = max(0.0, thr.low - va)
            e1 = max(0.0, thr.low - vb)
            excess = max(e0, e1)
            value = vb if e1 >= e0 else va  # 下限方向取更小值
            pieces.append(
                _Piece(
                    dim=dim,
                    start=a.ts + dt.timedelta(seconds=x0),
                    end=a.ts + dt.timedelta(seconds=x1),
                    dose=(e0 + e1) * 0.5 * width,
                    excess=excess,
                    value=value,
                    direction=LOW,
                )
            )
    return pieces


def compute_excursions(
    samples: list[Sample],
    temp_thresholds: Thresholds,
    hum_thresholds: Thresholds,
    max_gap_seconds: float,
) -> list[ExcursionEvent]:
    """按时间升序扫描相邻读数，生成合并后的越界事件列表。

    ``samples`` 需已按时间戳升序、无重复时间戳（调用方负责过滤时间
    窗口与去重）。间隔严格大于 ``max_gap_seconds`` 的读数对直接跳过，
    未知区间既不插值也不参与事件合并。
    """
    dim_thresholds = (
        (TEMPERATURE, temp_thresholds),
        (HUMIDITY, hum_thresholds),
    )

    pieces: list[_Piece] = []
    for a, b in zip(samples, samples[1:]):
        gap = (b.ts - a.ts).total_seconds()
        if gap <= 0.0 or gap > max_gap_seconds:
            # 重复时间戳不应出现（入库已合并）；超 gap 必须断开
            continue
        for dim, thr in dim_thresholds:
            if thr.enabled():
                pieces.extend(_dimension_pieces(a, b, gap, dim, thr))

    # 稳定排序：开始时间相同则温度优先、再按结束时间，保证可重放
    pieces.sort(key=lambda p: (p.start, _DIM_RANK[p.dim], p.end))

    events: list[_Accumulator] = []
    for piece in pieces:
        if events and piece.start <= events[-1].end:
            # 与上一事件相交或首尾相接（含某维度恰好于阈值点交接）：合并
            events[-1].merge(piece)
        else:
            events.append(_Accumulator.from_piece(piece))

    return [acc.finalize(i) for i, acc in enumerate(events)]

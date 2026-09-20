"""越界剂量算法单元测试：精确插值积分、max_gap 断开、事件合并与排序。"""

from __future__ import annotations

import datetime as dt

import pytest

from app.excursions import (
    HUMIDITY,
    TEMPERATURE,
    Thresholds,
    compute_excursions,
)
from app.segmentation import Sample

T0 = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


def at(minutes: float) -> dt.datetime:
    return T0 + dt.timedelta(minutes=minutes)


def sample(ts: dt.datetime, t: float, h: float) -> Sample:
    return Sample(ts=ts, temperature=t, humidity=h)


# ---------- 单维度上限：精确交点与积分 ----------


def test_high_threshold_triangle_trapezoid_dose():
    # 20→22→22→20，上限 21；每段 10 分钟
    samples = [sample(at(0), 20, 50), sample(at(10), 22, 50),
               sample(at(20), 22, 50), sample(at(30), 20, 50)]
    events = compute_excursions(samples, Thresholds(high=21.0), Thresholds(), 3600)
    assert len(events) == 1
    e = events[0]
    # 交点恰好在第 5、25 分钟
    assert e.start_ts == at(5)
    assert e.end_ts == at(25)
    assert e.duration_seconds == pytest.approx(1200.0)
    assert e.dimensions == [TEMPERATURE]
    # 上升三角形 300 + 平顶矩形 600 + 下降三角形 0... 实际：
    # 段1 三角形 (0+1)/2*300=150；段2 矩形 1*600=600；段3 三角形 150 → 900
    assert e.temperature_dose == pytest.approx(900.0)
    assert e.humidity_dose == 0.0
    assert e.peak_temperature == 22.0
    assert e.temperature_peak_excess == pytest.approx(1.0)
    assert e.peak_humidity is None
    assert e.humidity_peak_excess == 0.0


def test_low_threshold_dose():
    # 20→18→20，下限 19；每段 600 秒
    samples = [sample(at(0), 20, 50), sample(at(10), 18, 50), sample(at(20), 20, 50)]
    events = compute_excursions(samples, Thresholds(low=19.0), Thresholds(), 3600)
    assert len(events) == 1
    e = events[0]
    assert e.start_ts == at(5)
    assert e.end_ts == at(15)
    # 两个三角形：各 (0+1)/2*300 = 150
    assert e.temperature_dose == pytest.approx(300.0)
    assert e.peak_temperature == 18.0
    assert e.temperature_peak_excess == pytest.approx(1.0)


def test_exact_threshold_touch_gives_zero_dose():
    # 恰好触及阈值但从未越过：无事件
    samples = [sample(at(0), 20, 50), sample(at(10), 21, 50), sample(at(20), 20, 50)]
    events = compute_excursions(samples, Thresholds(high=21.0), Thresholds(), 3600)
    assert events == []


def test_threshold_within_excursion_is_not_subdivided_when_only_one_side():
    # 只配置上限 20：20→22→21 全程在上限之上（末点 21 仍 > 20），单段不切开
    samples = [sample(at(0), 20, 50), sample(at(10), 22, 50), sample(at(20), 21, 50)]
    events = compute_excursions(samples, Thresholds(high=20.0), Thresholds(), 3600)
    assert len(events) == 1
    e = events[0]
    assert e.start_ts == at(0)
    assert e.end_ts == at(20)
    # 段1：(0+2)/2*600=600；段2：(2+1)/2*600=900
    assert e.temperature_dose == pytest.approx(1500.0)


def test_high_and_low_both_configured_same_segment():
    # 同时配置上下限，区间内从高于上限穿到低于下限：方向分段且剂量分别累计
    samples = [sample(at(0), 22, 50), sample(at(10), 18, 50)]
    events = compute_excursions(
        samples, Thresholds(low=19.0, high=21.0), Thresholds(), 3600
    )
    # 22→18 线性：0~2.5min 越上限，7.5~10min 越下限，中间合规，
    # 两个子区间不相交 → 两个事件
    assert len(events) == 2
    first, second = events
    assert first.start_ts == at(0)
    assert first.end_ts == at(2.5)
    assert first.temperature_dose == pytest.approx((1 + 0) / 2 * 150)
    assert second.start_ts == at(7.5)
    assert second.end_ts == at(10)
    assert second.temperature_dose == pytest.approx((0 + 1) / 2 * 150)
    assert first.index == 0 and second.index == 1


# ---------- max_gap：断开与未知区间 ----------


def test_gap_exceeding_max_gap_splits_events():
    # 0~10 越限；70~80 越限；中间 60 分钟空档 > 30 分钟 max_gap
    samples = [sample(at(0), 20, 50), sample(at(10), 22, 50),
               sample(at(70), 22, 50), sample(at(80), 20, 50)]
    events = compute_excursions(samples, Thresholds(high=21.0), Thresholds(), 1800)
    assert len(events) == 2
    assert events[0].end_ts == at(10)
    assert events[1].start_ts == at(70)
    # 未知区间（10~70）不插值、不计剂量
    assert events[0].temperature_dose == pytest.approx(150.0)
    assert events[1].temperature_dose == pytest.approx(150.0)


def test_gap_equal_to_max_gap_is_continuous():
    # 间隔恰好等于 max_gap（60 分钟）仍在两点间插值；越限段被跨越，合并为一个事件
    samples = [sample(at(0), 22, 50), sample(at(10), 22, 50),
               sample(at(70), 22, 50), sample(at(80), 22, 50)]
    events = compute_excursions(samples, Thresholds(high=21.0), Thresholds(), 3600)
    assert len(events) == 1
    e = events[0]
    assert e.start_ts == at(0)
    assert e.end_ts == at(80)
    # 10~70 的空档也在“连续”范围内（值恒定 22），剂量含 60 分钟矩形
    assert e.temperature_dose == pytest.approx(1.0 * 80 * 60)


def test_unknown_interval_never_counts_even_if_linear_extrapolation_would_exceed():
    # gap 两端都越限；超 gap 时绝不在未知区间插值
    samples = [sample(at(0), 22, 50), sample(at(10), 22, 50),
               sample(at(70), 22, 50), sample(at(80), 22, 50)]
    events = compute_excursions(samples, Thresholds(high=21.0), Thresholds(), 1800)
    assert len(events) == 2
    # 每个事件仅覆盖自己的 10 分钟已知区间，而非跨空档合并
    assert events[0].start_ts == at(0) and events[0].end_ts == at(10)
    assert events[1].start_ts == at(70) and events[1].end_ts == at(80)
    assert sum(e.temperature_dose for e in events) == pytest.approx(2 * 600.0)


# ---------- 温湿联合事件合并 ----------


def test_overlapping_temperature_and_humidity_merge_into_one_event():
    # 温度 0~10 越上限（20→22→20），湿度 5~15 越上限（60→62→60）
    samples = [
        sample(at(0), 20, 60),
        sample(at(5), 21, 61),
        sample(at(10), 22, 62),
        sample(at(15), 20, 60),
    ]
    events = compute_excursions(
        samples, Thresholds(high=20.5), Thresholds(high=60.5), 3600
    )
    assert len(events) == 1
    e = events[0]
    assert e.dimensions == [TEMPERATURE, HUMIDITY]
    # 起始 = 温度交点（2.5min），结束 = 湿度交点（10+30/60*... 用近似）
    assert e.start_ts == at(2.5)
    # 湿度 60→61→62→60：交点在 60.5 上升段(2.5@0-5) 与下降段 10~15
    assert e.duration_seconds > 0
    assert e.temperature_dose > 0
    assert e.humidity_dose > 0
    assert e.peak_temperature == 22.0
    assert e.peak_humidity == 62.0


def test_touching_dimensions_merge_when_share_threshold_instant():
    # 温度越限 [0,5]（22→恰好 21 交点），湿度越限 [5,10]（61→62 段恰在 5 起越）：
    # 两区间在第 5 分钟首尾相接 → 合并为一个事件
    samples = [
        sample(at(0), 22, 60),
        sample(at(5), 21, 61),   # 温度恰好在此时刻回到上限；湿度恰在此时刻到达上限
        sample(at(10), 20, 62),
    ]
    # 温度上限 21：仅 0→5 越限；湿度上限 61：仅 5→10 越限
    events = compute_excursions(
        samples, Thresholds(high=21.0), Thresholds(high=61.0), 3600
    )
    assert len(events) == 1
    e = events[0]
    assert e.dimensions == [TEMPERATURE, HUMIDITY]
    assert e.start_ts == at(0)
    assert e.end_ts == at(10)
    assert e.duration_seconds == pytest.approx(600.0)
    # 温度三角形 (1+0)/2*300=150；湿度 (0+1)/2*300=150
    assert e.temperature_dose == pytest.approx(150.0)
    assert e.humidity_dose == pytest.approx(150.0)


def test_separate_dimension_excursions_stay_separate():
    # 温度越限在 0~5，湿度越限在 20~25，中间合规：两个独立事件
    samples = [
        sample(at(0), 22, 50),
        sample(at(10), 20, 50),
        sample(at(20), 20, 70),
        sample(at(30), 20, 50),
    ]
    events = compute_excursions(
        samples, Thresholds(high=21.0), Thresholds(high=60.0), 3600
    )
    assert len(events) == 2
    assert events[0].dimensions == [TEMPERATURE]
    assert events[1].dimensions == [HUMIDITY]
    # 按开始时间稳定排序
    assert events[0].start_ts < events[1].start_ts


def test_events_sorted_by_start_time_stable():
    # 三个互不相交的越限段，输入顺序在时间上本身有序；打乱样本顺序不应影响输出
    base = [
        sample(at(0), 22, 50), sample(at(10), 20, 50),
        sample(at(30), 22, 50), sample(at(40), 20, 50),
        sample(at(60), 22, 50), sample(at(70), 20, 50),
    ]
    events = compute_excursions(base, Thresholds(high=21.0), Thresholds(), 3600)
    starts = [e.start_ts for e in events]
    assert starts == sorted(starts)
    assert len(starts) == 3
    assert [e.index for e in events] == [0, 1, 2]


def test_disabled_dimension_produces_nothing():
    samples = [sample(at(0), 100, 0), sample(at(10), 100, 0)]
    events = compute_excursions(samples, Thresholds(), Thresholds(), 3600)
    assert events == []


def test_single_sample_and_empty():
    assert compute_excursions([sample(at(0), 22, 50)],
                              Thresholds(high=21.0), Thresholds(), 3600) == []
    assert compute_excursions([], Thresholds(high=21.0), Thresholds(), 3600) == []

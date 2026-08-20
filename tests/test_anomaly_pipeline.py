import joblib
import numpy as np
import pandas as pd
import pytest

from src.anomaly_features import ANOMALY_SENSOR_COLUMNS
from src.anomaly_pipeline import (
    AnomalyModelBundle,
    FROZEN_ISOLATION_FOREST_HYPERPARAMETERS,
    build_anomaly_detector,
    empirical_percentile,
    evaluate_anomaly_trajectory,
    evaluate_persistence,
    fit_anomaly_bundle,
    native_to_anomaly_score,
    score_anomaly_observation,
    sensor_deviation_context,
    threshold_exceeded,
)


def make_reference(rows=300):
    rng=np.random.default_rng(42)
    return pd.DataFrame(rng.normal(size=(rows,14)),columns=ANOMALY_SENSOR_COLUMNS)


@pytest.fixture(scope="module")
def bundle(): return fit_anomaly_bundle(make_reference())


def make_trajectory(length=5):
    reference=make_reference(length)
    reference.insert(0,"cycle",np.arange(1,length+1))
    return reference


def test_detector_hyperparameters_are_frozen():
    detector=build_anomaly_detector()
    assert all(detector.get_params()[key]==value for key,value in FROZEN_ISOLATION_FOREST_HYPERPARAMETERS.items())


def test_score_direction_and_observation_contract(bundle):
    observation=make_reference(1)
    result=score_anomaly_observation(bundle,observation)
    direct=-bundle.detector.score_samples(bundle.scaler.transform(observation))[0]
    assert result["current_anomaly_score"]==pytest.approx(direct)
    assert np.isfinite(result["current_anomaly_score"])
    assert 0<=result["anomaly_percentile"]<=100
    assert "probability" not in result


def test_empirical_percentile_edges_ties_and_monotonicity():
    reference=np.array([1.,2.,2.,4.])
    mapped=empirical_percentile([-1,1,2,3,5],reference)
    np.testing.assert_allclose(mapped,[0,25,75,75,100])
    assert np.all(np.diff(mapped)>=0)


def test_threshold_equality_is_not_exceeded():
    np.testing.assert_array_equal(threshold_exceeded([.4,.5,.6],.5),[False,False,True])


@pytest.mark.parametrize("length",[1,2,3,4])
def test_short_history_never_activates(bundle,length):
    result=evaluate_anomaly_trajectory(bundle,make_trajectory(length))
    assert result["persistence_status"]=="insufficient_history"
    assert result["alert_active"] is False
    assert result["recent_window_size"]==length


@pytest.mark.parametrize("pattern,active",[
    ([True,True,True,False,False],True),
    ([True,True,False,False,False],False),
    ([True,True,True,True,False],True),
])
def test_three_of_five_persistence(pattern,active):
    result=evaluate_persistence(pattern)
    assert result["persistence_status"]=="available"
    assert result["alert_active"] is active


def test_only_latest_five_affect_persistence():
    assert evaluate_persistence([True]*10+[False]*5)["alert_active"] is False
    assert evaluate_persistence([False]*10+[True,True,True,False,False])["alert_active"] is True


def test_sensor_deviation_formula_direction_order_and_top_n(bundle):
    values=bundle.scaler.mean_.copy()
    values[0]+=3*bundle.scaler.scale_[0]
    values[1]-=2*bundle.scaler.scale_[1]
    observation=pd.DataFrame([values],columns=ANOMALY_SENSOR_COLUMNS)
    context=sensor_deviation_context(bundle,observation,top_n=2)
    assert [item["sensor"] for item in context]==ANOMALY_SENSOR_COLUMNS[:2]
    assert context[0]["standardized_deviation"]==pytest.approx(3)
    assert context[0]["direction"]=="above_normal"
    assert context[1]["standardized_deviation"]==pytest.approx(-2)
    assert context[1]["direction"]=="below_normal"


def test_serialized_bundle_reproduces_observation_and_trajectory(bundle,tmp_path):
    observation=make_reference(1); trajectory=make_trajectory(5)
    expected_observation=score_anomaly_observation(bundle,observation)
    expected_trajectory=evaluate_anomaly_trajectory(bundle,trajectory)
    path=tmp_path/"anomaly.joblib"; joblib.dump(bundle,path); loaded=joblib.load(path)
    assert isinstance(loaded,AnomalyModelBundle)
    assert score_anomaly_observation(loaded,observation)==expected_observation
    assert evaluate_anomaly_trajectory(loaded,trajectory)==expected_trajectory


def test_notebook_formula_parity(bundle):
    observation=make_reference(1)
    standardized=(observation.to_numpy()-bundle.scaler.mean_)/bundle.scaler.scale_
    np.testing.assert_allclose(bundle.scaler.transform(observation),standardized)
    direct=native_to_anomaly_score(bundle.detector.score_samples(standardized))[0]
    result=score_anomaly_observation(bundle,observation)
    assert result["current_anomaly_score"]==pytest.approx(direct)
    expected=100*np.searchsorted(bundle.sorted_normal_scores,direct,side="right")/len(bundle.sorted_normal_scores)
    assert result["anomaly_percentile"]==pytest.approx(expected)
    assert bundle.raw_threshold==pytest.approx(np.quantile(bundle.sorted_normal_scores,.975))

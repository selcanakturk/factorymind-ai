import numpy as np
import pandas as pd
import pytest

from src.anomaly_features import (
    ANOMALY_PREDICTOR_COUNT,
    ANOMALY_SENSOR_COLUMNS,
    validate_anomaly_observation,
    validate_anomaly_trajectory,
)


EXPECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def make_observation(offset=0.0):
    return pd.DataFrame([{sensor: index + offset for index, sensor in enumerate(EXPECTED_SENSORS, 1)}])


def make_trajectory(length=5, unit_id=1):
    rows=[]
    for cycle in range(1, length+1):
        row={"unit_id":unit_id,"cycle":cycle}
        row.update({sensor:index+cycle/10 for index,sensor in enumerate(EXPECTED_SENSORS,1)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_sensor_contract():
    assert ANOMALY_SENSOR_COLUMNS == EXPECTED_SENSORS
    assert ANOMALY_PREDICTOR_COUNT == 14
    assert not {"unit_id","cycle","raw_rul","RUL","target","failure"}.intersection(ANOMALY_SENSOR_COLUMNS)


def test_valid_observation_is_accepted_in_frozen_order():
    validated=validate_anomaly_observation(make_observation())
    assert list(validated.columns)==EXPECTED_SENSORS and validated.shape==(1,14)


@pytest.mark.parametrize("mutator,message",[
    (lambda x:x.drop(columns="sensor_2"),"missing"),
    (lambda x:x.assign(extra=1),"extra"),
    (lambda x:x.assign(raw_rul=100),"not accepted"),
    (lambda x:x.assign(RUL=100),"not accepted"),
    (lambda x:x.assign(sensor_2=np.nan),"finite"),
    (lambda x:x.assign(sensor_2=np.inf),"finite"),
    (lambda x:x.assign(sensor_2=True),"boolean"),
    (lambda x:x.assign(sensor_2="1.0"),"numeric strings"),
])
def test_invalid_observations_are_rejected(mutator,message):
    with pytest.raises(ValueError,match=message): validate_anomaly_observation(mutator(make_observation()))


@pytest.mark.parametrize("length",[1,4,5])
def test_supported_trajectory_lengths(length):
    assert validate_anomaly_trajectory(make_trajectory(length)).shape==(length,14)


@pytest.mark.parametrize("mutator,message",[
    (lambda x:pd.concat([x,x.iloc[[-1]]],ignore_index=True),"Duplicate cycles"),
    (lambda x:x.drop(index=2).reset_index(drop=True),"consecutive"),
    (lambda x:x.iloc[::-1].reset_index(drop=True),"increasing"),
    (lambda x:x.assign(cycle=[0,2,3,4,5]),"positive"),
    (lambda x:x.assign(cycle=[1,2,3.5,4,5]),"integers"),
    (lambda x:x.assign(unit_id=[1,1,2,1,1]),"exactly one"),
])
def test_invalid_trajectories_are_rejected(mutator,message):
    with pytest.raises(ValueError,match=message): validate_anomaly_trajectory(mutator(make_trajectory()))


def test_optional_unit_id_and_missing_identity_behavior():
    validate_anomaly_trajectory(make_trajectory().drop(columns="unit_id"))
    bad=make_trajectory(); bad.loc[2,"unit_id"]=np.nan
    with pytest.raises(ValueError,match="cannot be missing"): validate_anomaly_trajectory(bad)

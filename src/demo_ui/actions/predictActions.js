import {
  PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE, STATUS, STATUS_FAILURE, STATUS_SUCCESS,
} from '../constants/actions';

export function predict() {
  return {
    type: PREDICT,
  };
}

export function predictSuccess(data) {
  return {
    type: PREDICT_SUCCESS,
    numPredictions: data.numPredictions,
    data: data.data,
  };
}

export function predictFailure(errorCode) {
  return {
    type: PREDICT_FAILURE,
    errorCode,
  };
}

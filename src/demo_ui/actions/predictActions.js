import { CLEAR_ERROR, PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE } from '../constants/actions';

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

export function predictFailure(error) {
  return {
    type: PREDICT_FAILURE,
    error: error.response.data,
  };
}

export function clearError() {
  return {
    type: CLEAR_ERROR,
  };
}

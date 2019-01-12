import { PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE, CLEAR_ERROR } from '../constants/actions';

const predictions = (state = {}, action) => {
  switch (action.type) {
    case CLEAR_ERROR:
      return Object.assign({}, state, {
        error: null,
      });
    case PREDICT:
      return Object.assign({}, state, {
        loading: true,
        error: null,
      });
    case PREDICT_SUCCESS:
      return Object.assign({}, state, {
        loading: false,
        bestAnswer: action.bestAnswer,
        numPredictions: action.numPredictions,
        data: action.data,
        error: null,
      });
    case PREDICT_FAILURE:
      return Object.assign({}, state, {
        loading: false,
        error: action.error,
      });
    default:
      return state;
  }
};

export default predictions;

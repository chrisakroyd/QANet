import { PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE } from '../constants/actions';

const predictions = (state = {}, action) => {
  switch (action.type) {
    case PREDICT:
      return Object.assign({}, state, {
        loading: true,
      });
    case PREDICT_SUCCESS:
      return Object.assign({}, state, {
        loading: false,
        numPredictions: action.numPredictions,
        data: action.data,
      });
    case PREDICT_FAILURE:
      return Object.assign({}, state, {
        loading: false,
        errorCode: action.errorCode,
      });
    default:
      return state;
  }
};

export default predictions;

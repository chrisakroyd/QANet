import { PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE } from '../constants/actions';

const predictions = (state = {}, action) => {
  switch (action.type) {
    case PREDICT:
      return {
        text: state.text,
        processed: '',
        attentionWeights: [],
        classification: '',
        confidence: 0.0,
        loading: true,
      };
    case PREDICT_SUCCESS:
      return {
        text: state.text,
        processed: action.data[0].processed,
        attentionWeights: action.data[0].attentionWeights,
        classification: action.data[0].classification,
        confidence: action.data[0].confidence,
        loading: false,
      };
    case PREDICT_FAILURE:
      return {
        text: action.text,
        processed: state.processed,
        attentionWeights: state.attentionWeights,
        classification: state.classification,
        confidence: state.confidence,
        loading: false,
        errorCode: action.error_code,
      };
    default:
      return state;
  }
};

export default predictions;

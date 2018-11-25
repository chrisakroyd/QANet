const text = {
  loading: false,
  loadContextFromUrl: false,
  query: '',
  context: '',
  contextUrl: '',
  error: null,
};

const predictions = {
  loading: false,
  numPredictions: 0,
  bestAnswer: '',
  data: [],
  error: null,
};

export {
  text,
  predictions,
};

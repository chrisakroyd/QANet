const text = {
  loading: false,
  loadContextFromUrl: false,
  query: '',
  context: '',
  contextUrl: '',
  errorCode: -1,
};

const predictions = {
  loading: false,
  query: '',
  numAnswers: 0,
  contextTokens: [],
  queryTokens: [],
  answerTexts: [],
  answerStarts: [],
  answerEnds: [],
  startProb: [],
  endProb: [],
  errorCode: -1,
};

export {
  text,
  predictions,
};

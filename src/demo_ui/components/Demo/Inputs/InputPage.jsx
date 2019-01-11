import React from 'react';
import PropTypes from 'prop-types';

import ContextOptions from '../../common/ContextOptions';
import InputBar from '../../common/InputBar';
import InputBox from '../../common/InputBox';
import Step from '../../common/Step';
import Predict from './Predict';
import { predictShape, textShape } from '../../../prop-shapes/index';

const InputPage = ({
  predict, setQueryText, setContextText, enterText, loadExample, text, predictions,
}) => (
  <div className="input-page">
    <div className="section">
      <div className="section-header">
        <Step number={1} />
        <div className="header-text">Enter a context</div>
      </div>
      <ContextOptions
        enterTextFunc={enterText}
        exampleFunc={loadExample}
        loadExample={text.loadExample}
      />
      <InputBox
        placeholder="Context"
        value={text.context}
        onKeyPress={setContextText}
        validInput={text.context.length > 0}
      />
    </div>
    <div className="section">
      <div className="section-header">
        <Step number={2} />
        <div className="header-text">Ask a question</div>
      </div>
      <InputBar
        placeholder="Question"
        value={text.query}
        onKeyPress={setQueryText}
        validInput={text.query.length > 0}
      />
    </div>
    <div className="section">
      <div className="section-header">
        <Step number={3} />
        <div className="header-text">Get an Answer</div>
      </div>
      <Predict
        predict={predict}
        loading={predictions.loading || text.loading}
        error={predictions.error}
        text={text}
      />
    </div>
  </div>
);

InputPage.propTypes = {
  predict: PropTypes.func.isRequired,
  setQueryText: PropTypes.func.isRequired,
  setContextText: PropTypes.func.isRequired,
  enterText: PropTypes.func.isRequired,
  loadExample: PropTypes.func.isRequired,
  text: textShape.isRequired,
  predictions: predictShape.isRequired,
};

export default InputPage;

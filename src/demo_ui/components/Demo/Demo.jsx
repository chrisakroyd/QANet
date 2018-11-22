import React from 'react';
import PropTypes from 'prop-types';

import ContextOptions from './ContextOptions/ContextOptions';
import Introduction from './Introduction/Introduction'
import InputBar from './InputBar/InputBar';
import InputBox from './InputBox/InputBox';
import PredictButton from './PredictButton/PredictButton';
import WordHeat from './WordHeat/WordHeat';
import Step from './Step/Step';
import predictShape from './../../prop-shapes/predictShape';
import textShape from './../../prop-shapes/textShape';
// import { predictShape, textShape } from './../../prop-shapes';
import './demo.scss';

const Demo = ({
  predict, setQueryText, setContextText, setContextUrl, text, predictions,
}) => {
  return (
    <div className="demo-body">
      <div className="section">
        <Introduction />
      </div>
      <div className="section">
        <Step number={1} label="Ask a question" />
        <InputBar
          value={text.query}
          onKeyPress={setQueryText}
        />
      </div>
      <div className="section">
        <Step number={2} label="Enter a context" />
        <ContextOptions onClick={setContextUrl} useUrl={text.loadContextFromUrl} />
        <InputBox
          value={text.context}
          onKeyPress={setContextText}
        />
      </div>
      <div className="section">
        <Step number={3} label="Get an Answer" />
        <div className="button-container">
          <PredictButton onEnter={predict} />
        </div>
      </div>
    </div>
  );
};

Demo.propTypes = {
  // Functions
  predict: PropTypes.func.isRequired,
  setQueryText: PropTypes.func.isRequired,
  setContextText: PropTypes.func.isRequired,
  setContextUrl: PropTypes.func.isRequired,
  // Data
  text: textShape.isRequired,
  predictions: predictShape.isRequired,
};

export default Demo;

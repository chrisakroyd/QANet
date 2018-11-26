import React from 'react';
import PropTypes from 'prop-types';

import ContextOptions from './ContextOptions/ContextOptions';
import InputBar from './InputBar/InputBar';
import InputBox from './InputBox/InputBox';
import Button from '../common/Button';
import Step from './Step/Step';
import textShape from '../../prop-shapes/textShape';


const InputPage = ({
  predict, setQueryText, setContextText, enterText, loadExample, text,
}) => (
  <div className="input-page">
    <div className="section">
      <Step number={1} label="Ask a question" />
      <InputBar placeholder="Question" value={text.query} onKeyPress={setQueryText} />
    </div>
    <div className="section">
      <Step number={2} label="Enter a context" />
      <ContextOptions enterTextFunc={enterText} exampleFunc={loadExample} loadExample={text.loadExample} />
      <InputBox placeholder="Context" value={text.context} onKeyPress={setContextText} />
    </div>
    <div className="section">
      <Step number={3} label="Get an Answer" />
      <Button onClick={predict} label="Predict" />
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
};

export default InputPage;

import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';
import { predictShape } from '../../../prop-shapes';
import { capitalizeFirstLetter, isError } from '../../../util';
import { pointerPredTip } from '../../../constants/tooltips';

import BackButton from '../../common/BackButton';
import LoadingSpinner from '../../common/LoadingSpinner';
import Step from '../../common/Step';
import ToolTip from '../../common/ToolTip';
import WordHeat from './WordHeat';


const ResultsPage = ({ goBack, query, predictions }) => {
  let answerContent;

  // If we somehow end up on this page without anything to display go back (e.g. Loading by URL)
  // unless we have an error.
  if ((predictions.numPredictions <= 0 && !predictions.loading) && !isError(predictions.error)) {
    goBack();
  }

  if (predictions.loading) {
    answerContent = (
      <div className="section spinner-container">
        <LoadingSpinner />
      </div>
    );
  } else {
    const pointerHeatmaps = predictions.data.map(prediction => (
      <WordHeat
        key={shortid.generate()}
        words={prediction.contextTokens}
        startProb={prediction.startProb}
        endProb={prediction.endProb}
        answerStart={prediction.answerStart}
        answerEnd={prediction.answerEnd}
      />));

    const sortedAnswers = predictions.data.sort((a, b) => b.answerProb - a.answerProb);
    const topAnswers = sortedAnswers.map((prediction, number) => (
      <div className="answer" key={shortid.generate()}>
        <Step number={number + 1} small />
        <div className="answer-text">{capitalizeFirstLetter(prediction.answerText)}</div>
      </div>
    ));

    answerContent = (
      <div>
        <div className="section">
          <div className="header-text">Top Answers</div>
          <div className="answers-container">
            {topAnswers}
          </div>
        </div>
        <div className="section">
          <div className="section-header">
            <div className="header-text">Pointer predictions</div>
            <ToolTip tip={pointerPredTip} />
          </div>
          <div>
            {pointerHeatmaps}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="section">
        <div className="answer-header flex-grid">
          <div className="flex-grid-third">
            <BackButton onClick={goBack} />
          </div>
          <div className="flex-grid-third header-text">Results</div>
          <div className="flex-grid-third"></div>
        </div>

      </div>
      <div className="section">
        <div className="header-text">Your question</div>
        <p className="query">{query}</p>
      </div>
      {answerContent}
    </div>
  );
};

ResultsPage.propTypes = {
  goBack: PropTypes.func.isRequired,
  query: PropTypes.string.isRequired,
  predictions: predictShape.isRequired,
};

export default ResultsPage;

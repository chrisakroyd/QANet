import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';
import predictShape from './../../prop-shapes/predictShape';
import { capitalizeFirstLetter, isError } from './../../util';


import BackButton from '../common/BackButton';
import WordHeat from './WordHeat/WordHeat';

import './results-page.scss';

const ResultsPage = ({ goBack, query, predictions }) => {
  let answerContent;

  // If we somehow end up on this page without anything to display go back (e.g. Loading by URL)
  // unless we have an error.
  if ((predictions.numPredictions <= 0 && !predictions.loading) && !isError(predictions.error)) {
    goBack();
  }

  if (predictions.loading) {
    answerContent = <div>Loading...</div>;
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


    answerContent = (
      <div>
        <div className="section">
          <div className="header-text">Best Answer</div>
          <div className="answer-container">
            <p>{capitalizeFirstLetter(predictions.bestAnswer)}</p>
          </div>
        </div>
        <div className="section">
          <div className="header-text">Pointer predictions</div>
          {pointerHeatmaps}
        </div>
        <div className="section">
          <div className="header-text">Probability Distribution</div>
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
          <div className="flex-grid-third header-text">Your Question</div>
          <div className="flex-grid-third"></div>
        </div>
        <div className="answer-container">
          <p>{query}</p>
        </div>
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

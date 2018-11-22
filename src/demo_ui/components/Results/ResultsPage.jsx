import React from 'react';
import PropTypes from 'prop-types';
import predictShape from './../../prop-shapes/predictShape';

const ResultsPage = ({ goBack, predictions }) => {
  return (
    <div>
      test
    </div>
  );
};

ResultsPage.propTypes = {
  goBack: PropTypes.func.isRequired,
  predictions: predictShape.isRequired,
};

export default ResultsPage;

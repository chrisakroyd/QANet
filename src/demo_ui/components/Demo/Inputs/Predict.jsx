import React from 'react';
import PropTypes from 'prop-types';

import Button from '../../common/Button';
import LoadingSpinner from '../../common/LoadingSpinner';
import { errorShape, textShape } from '../../../prop-shapes';


const Predict = ({ text, loading, predict, error }) => {
  const hasQuery = text.query.length > 0;
  const hasContext = text.context.length > 0;
  const validInput = (hasQuery && hasContext) && error === null;
  let content = (
    <Button
      onClick={predict}
      label="Predict"
      enabled={validInput}
    />
  );

  if (!validInput && !loading) {
    let errorMessage = '';
    if (!hasQuery) {
      errorMessage = 'Please enter a valid query.';
    } else if (!hasContext) {
      errorMessage = 'Please enter a valid context.';
    } else if (error !== null) {
      errorMessage = error.errorMessage;
    }
    content = (
      <div className="predict-action">
        <Button
          onClick={() => {
          }}
          label="Predict"
          enabled={validInput}
        />
        <p className="error-message">Error: {errorMessage}</p>
      </div>
    );
  } else if (loading) {
    content = (<LoadingSpinner />);
  }

  return (
    <div className="predict-action">
      {content}
    </div>);
};

Predict.propTypes = {
  predict: PropTypes.func.isRequired,
  loading: PropTypes.bool.isRequired,
  text: textShape.isRequired,
  error: errorShape,
};

Predict.defaultProps = {
  error: null,
};

export default Predict;

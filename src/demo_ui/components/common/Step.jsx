import React from 'react';
import classnames from 'classnames';
import PropTypes from 'prop-types';

import './step.scss';

const Step = ({ number, small }) => (
  <div className={classnames('step', { 'small-step': small })}>{number}</div>
);

Step.propTypes = {
  number: PropTypes.number.isRequired,
  small: PropTypes.bool,
};

Step.defaultProps = {
  small: false,
};

export default Step;

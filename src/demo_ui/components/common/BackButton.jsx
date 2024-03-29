import React from 'react';
import PropTypes from 'prop-types';
import './inputs.scss';

/**
 * BackArrow() returns a clickable left facing arrow (SVG code from material design).
 */
const BackArrow = ({ onClick }) => (
  <div className="general-button back-button button" onClick={() => onClick()} onKeyPress={() => onClick()} role="button" tabIndex={0}>
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
      <path d="M0 0h24v24H0z" fill="none" />
      <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
    </svg>
    Back
  </div>
);

BackArrow.propTypes = {
  onClick: PropTypes.func.isRequired,
};

export default BackArrow;

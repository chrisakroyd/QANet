import React from 'react';
import PropTypes from 'prop-types';
import './tooltip.scss';

const ToolTip = ({ tip }) => (
  <div className="tool-tip">
    <div className="tip" tooltip={tip}>?</div>
  </div>
);

ToolTip.propTypes = {
  tip: PropTypes.string.isRequired,
};

export default ToolTip;

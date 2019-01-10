import React from 'react';
import PropTypes from 'prop-types';

import './inputs.scss';

class InputBar extends React.Component {
  render() {
    return (
      <div className="input-bar">
        <input
          onChange={() => this.props.onKeyPress(this.textInput.value)}
          placeholder={this.props.placeholder}
          value={this.props.value}
          ref={(input) => { this.textInput = input; }}
        />
      </div>
    );
  }
}

InputBar.propTypes = {
  onKeyPress: PropTypes.func.isRequired,
  placeholder: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
};

export default InputBar;

import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import './inputs.scss';


class InputBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      focused: false,
    };
  }

  onBlur() {
    this.setState({ focused: false });
  }

  onFocus() {
    this.setState({ focused: true });
  }

  render() {
    const classes = classNames('input-bar', {
      invalid: !this.props.validInput && this.state.focused,
      focused: this.state.focused && this.props.validInput,
    });

    return (
      <div className={classes}>
        <input
          onChange={() => this.props.onKeyPress(this.textInput.value)}
          onFocus={() => this.onFocus()}
          onBlur={() => this.onBlur()}
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
  validInput: PropTypes.bool,
};

InputBar.defaultProps = {
  validInput: true,
};

export default InputBar;

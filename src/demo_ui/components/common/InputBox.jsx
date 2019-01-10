import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import './inputs.scss';


class InputBox extends React.Component {
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
    const classes = classNames('input-box', {
      invalid: !this.props.validInput && this.state.focused,
      focused: this.state.focused && this.props.validInput,
    });

    return (
      <div className={classes}>
        <textarea
          className="text-area"
          onChange={() => this.props.onKeyPress(this.textInput.value)}
          onFocus={() => this.onFocus()}
          onBlur={() => this.onBlur()}
          value={this.props.value}
          ref={(input) => { this.textInput = input; }}
        />
      </div>
    );
  }
}

InputBox.propTypes = {
  value: PropTypes.string.isRequired,
  onKeyPress: PropTypes.func.isRequired,
  validInput: PropTypes.bool,
};

InputBox.defaultProps = {
  validInput: true,
};

export default InputBox;

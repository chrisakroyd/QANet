import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import shortid from 'shortid';

import { interpolateHcl } from 'd3-interpolate';

import './word-heat.scss';

const highColour = '#5fcf80';
const lowColour = '#fbfefc';
const interpolate = interpolateHcl(lowColour, highColour);

class WordHeat extends React.Component {
  generateWordComponents() {
    const {
      words, startProb, endProb, answerStart, answerEnd,
    } = this.props;
    return words.map((word, i) => {
      const wordClass = classNames('heat-word', { 'answer-segment': i >= answerStart && i <= answerEnd });
      let score;

      if (startProb[i] > endProb[i]) {
        score = startProb[i];
      } else {
        score = endProb[i];
      }

      return (
        <div
          key={shortid.generate()}
          className={wordClass}
          style={({ backgroundColor: interpolate(score) })}
        >
          {word}
        </div>);
    });
  }


  render() {
    return (
      <div className="word-heat">
        <div className="heat-word-list">
          {this.generateWordComponents()}
        </div>
      </div>
    );
  }
}

WordHeat.propTypes = {
  words: PropTypes.arrayOf(PropTypes.string).isRequired,
  startProb: PropTypes.arrayOf(PropTypes.number).isRequired,
  endProb: PropTypes.arrayOf(PropTypes.number).isRequired,
  answerStart: PropTypes.number.isRequired,
  answerEnd: PropTypes.number.isRequired,
};

export default WordHeat;

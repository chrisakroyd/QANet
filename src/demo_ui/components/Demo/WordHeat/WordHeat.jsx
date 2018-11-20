import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';

import * as d3 from 'd3';
import './word-heat.scss';

const highColour = '#ff6d77';
// const lowColour = '#fe9f60';
const lowColour = '#f2e5f1';

const interpolate = d3.interpolateHcl(lowColour, highColour);

class WordHeat extends React.Component {
  generateWordComponents() {
    const splitWords = this.props.words.split(' ');
    const scores = this.props.scores;

    return splitWords.map((word, i) =>
      (
        <li
          key={shortid.generate()}
          className="heat-word"
          style={({ backgroundColor: interpolate(scores[i] * 15) })}
        >
          {word}
        </li>));
  }


  render() {
    return (
      <div className="word-heat">
        <ul className="heat-word-list">
          {this.generateWordComponents()}
        </ul>
      </div>
    );
  }
}

WordHeat.propTypes = {
  words: PropTypes.string.isRequired,
  scores: PropTypes.arrayOf(PropTypes.number).isRequired,
};

export default WordHeat;

import React from 'react';
import PropTypes from 'prop-types';
import * as d3 from 'd3';

import './graph.scss';

const width = 750;
const height = 200;
const margin = {
  top: 5,
  right: 0,
  bottom: 30,
  left: 35,
};


class ProbabilityVisualiser extends React.Component {
  componentDidMount() {
    const { data, max } = this.dataAndStats();
    const node = this.node;

    const zoom = d3.zoom()
      .scaleExtent([1, 5])
      .translateExtent([[0, 0], [width, height]])
      .on('zoom', zoomed);

    const xScale = d3.scaleLinear()
      .range([0, (width - margin.left - margin.right)])
      .domain([0, data.length]);

    const yScale = d3.scaleLinear()
      .range([(height - margin.top - margin.bottom), 0])
      .domain([0.0, max]);

    const svg = d3.select(node)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'shadow')
      .call(zoom)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Stop lines appearing outside of axis.
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', width)
      // 6 is for height of the axis.
      .attr('height', (height - margin.bottom - 4));


    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);
    // line function convention (feeds an array)
    const startProbLine = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScale(d.startProb))
      .curve(d3.curveMonotoneX);

    const endProbLine = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScale(d.endProb))
      .curve(d3.curveMonotoneX);

    const q2cLine = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScale(d.q2c))
      .curve(d3.curveMonotoneX);

    // Append y axis + label
    svg.append('g')
      .attr('class', 'y axis')
      .call(yAxis);

    // Append the x axis + label
    svg.append('g')
      .attr('class', 'x axis')
      .attr('transform', `translate(0, ${(height - margin.top - margin.bottom)})`)
      .call(xAxis);
    svg.append('text')
      .attr('class', 'axis-label')
      .attr('transform', `translate(${width / 2}, ${height - margin.top})`)
      .text('Word Position');

    const startPlotted = svg.append('path')
      .datum(data)
      .attr('class', 'line prob-start')
      .attr('d', startProbLine);

    const endPlotted = svg.append('path')
      .datum(data)
      .attr('class', 'line prob-end')
      .attr('d', endProbLine);
    //
    const c2qPlotted = svg.append('path')
      .datum(data)
      .attr('class', 'line q2c')
      .attr('d', q2cLine);

    function zoomed() {
      // Update Scales
      const newYScale = d3.event.transform.rescaleY(yScale);
      const newXScale = d3.event.transform.rescaleX(xScale);
      // re-scale axes and gridlines during zoom
      svg.select(".y.axis").transition()
        .duration(50)
        .call(yAxis.scale(newYScale));

      svg.select(".x.axis").transition()
        .duration(50)
        .call(xAxis.scale(newXScale));

      // re-draw line
      const startProbScaledLine = d3.line()
        .x((d, i) => newXScale(i))
        .y(d => newYScale(d.startProb))
        .curve(d3.curveMonotoneX);

      const endProbScaledLine = d3.line()
        .x((d, i) => newXScale(i))
        .y(d => newYScale(d.endProb))
        .curve(d3.curveMonotoneX);

      const c2qScaledLine = d3.line()
        .x((d, i) => newXScale(i))
        .y(d => newYScale(d.q2c))
        .curve(d3.curveMonotoneX);


      startPlotted.attr('d', startProbScaledLine);
      endPlotted.attr('d', endProbScaledLine);
      c2qPlotted.attr('d', c2qScaledLine);
    }
  }

  dataAndStats() {
    const {
      words, startProb, endProb, q2c,
    } = this.props;
    const data = [];
    let max = 0;

    for (let i = 0; i < words.length; i++) {
      const posMax = Math.max(startProb[i], endProb[i], q2c[i]);
      if (posMax > max) {
        max = posMax;
      }

      data.push({
        word: words[i],
        startProb: startProb[i],
        endProb: endProb[i],
        q2c: q2c[i],
      });
    }
    return { data, max };
  }


  render() {
    return (
      <div
        className="probability-graph"
        ref={(node) => { this.node = node; }}
      />
    );
  }
}

ProbabilityVisualiser.propTypes = {
  words: PropTypes.arrayOf(PropTypes.string).isRequired,
  startProb: PropTypes.arrayOf(PropTypes.number).isRequired,
  endProb: PropTypes.arrayOf(PropTypes.number).isRequired,
  q2c: PropTypes.arrayOf(PropTypes.number).isRequired,
  answerStart: PropTypes.number.isRequired,
  answerEnd: PropTypes.number.isRequired,
};

export default ProbabilityVisualiser;

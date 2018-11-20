import React from 'react';
import createHashHistory from 'history/createHashHistory';
import thunk from 'redux-thunk';
import { render } from 'react-dom';
import { createStore, applyMiddleware, combineReducers } from 'redux';
import { routerMiddleware, routerReducer } from 'react-router-redux';

import searchApp from './reducers';
import Root from './components/Root';
import { predictions, text } from './constants/defaults';

import './index.scss';

let middleware;

const history = createHashHistory();


if (process.env.NODE_ENV !== 'production') {
  const { logger } = require('redux-logger');
  const { composeWithDevTools } = require('redux-devtools-extension');
  middleware = composeWithDevTools(applyMiddleware(
    routerMiddleware(history),
    thunk,
    logger,
  ));
} else {
  middleware = applyMiddleware(thunk);
}

const reducers = combineReducers({
  ...searchApp,
  router: routerReducer,
});

const defaultState = { predictions, text };

const store = createStore(
  reducers,
  defaultState,
  middleware,
);

render(
  <Root store={store} history={history} />,
  document.getElementById('app-container'),
);

# Copyright 2024 The TensorTrade and TensorTrade-NG Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from __future__ import annotations

import os
import typing

import plotly.graph_objects as go
from IPython.display import display, clear_output
from plotly.subplots import make_subplots

from tensortrade.env.interfaces import AbstractRenderer
from tensortrade.env.renderers.utils import check_valid_format, check_path, create_auto_file_name

if typing.TYPE_CHECKING:
    from typing import Tuple, Union
    import pandas as pd

    from collections import OrderedDict

class PlotlyTradingChart(AbstractRenderer):
    """Trading visualization for TensorTrade using Plotly.

    Parameters
    ----------
    display : bool
        True to display the chart on the screen, False for not.
    height : int
        Chart height in pixels. Affects both display and saved file
        charts. Set to None for 100% height. Default is None.
    save_format : str
        A format to save the chart to. Acceptable formats are
        html, png, jpeg, webp, svg, pdf, eps. All the formats except for
        'html' require Orca. Default is None for no saving.
    path : str
        The path to save the char to if save_format is not None. The folder
        will be created if not found.
    filename_prefix : str
        A string that precedes automatically-created file name
        when charts are saved. Default 'chart_'.
    timestamp_format : str
        The format of the date shown in the chart title.
    auto_open_html : bool
        Works for save_format='html' only. True to automatically
        open the saved chart HTML file in the default browser, False otherwise.
    include_plotlyjs : Union[bool, str]
        Whether to include/load the plotly.js library in the saved
        file. 'cdn' results in a smaller file by loading the library online but
        requires an Internet connect while True includes the library resulting
        in much larger file sizes. False to not include the library. For more
        details, refer to https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html

    Notes
    -----
    Possible Future Enhancements:
        - Saving images without using Orca.
        - Limit displayed step range for the case of a large number of steps and let
          the shown part of the chart slide after filling that range to keep showing
          recent data as it's being added.

    References
    ----------
    .. [1] https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html
    .. [2] https://plot.ly/python/figurewidget/
    .. [3] https://plot.ly/python/subplots/
    .. [4] https://plot.ly/python/reference/#candlestick
    .. [5] https://plot.ly/python/#chart-events
    """

    registered_name = "plotly_trading_chart"

    def __init__(self,
                 display: bool = True,
                 height: int = None,
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S',
                 save_format: str = None,
                 path: str = 'charts',
                 filename_prefix: str = 'chart_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._timestamp_format = timestamp_format
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html

        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

        self.fig = None
        self._price_chart = None
        self._volume_chart = None
        self._performance_chart = None
        self._net_worth_chart = None
        self._base_annotations = None
        self._last_trade_step = 0
        self._show_chart = display

    def _create_figure(self, performance_keys: dict) -> None:
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
        )
        fig.add_trace(go.Candlestick(name='Price', xaxis='x1', yaxis='y1',
                                     showlegend=False), row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.add_trace(go.Bar(name='Volume', showlegend=False,
                             marker={'color': 'DodgerBlue'}),
                      row=2, col=1)

        for k in performance_keys:
            fig.add_trace(go.Scatter(mode='lines', name=k), row=3, col=1)

        fig.add_trace(go.Scatter(mode='lines', name='Net Worth', marker={'color': 'DarkGreen'}),
                      row=4, col=1)

        fig.update_xaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_yaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_xaxes(title_text='Price', row=1)
        fig.update_xaxes(title_text='Volume', row=2)
        fig.update_xaxes(title_text='Performance', row=3)
        fig.update_xaxes(title_text='Net Worth', row=4)
        fig.update_xaxes(title_standoff=7, title_font=dict(size=12))

        self.fig = go.FigureWidget(fig)
        self._price_chart = self.fig.data[0]
        self._volume_chart = self.fig.data[1]
        self._performance_chart = self.fig.data[2]
        self._net_worth_chart = self.fig.data[-1]

        self.fig.update_annotations({'font': {'size': 12}})
        self.fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
        self._base_annotations = self.fig.layout.annotations

    def _create_trade_annotations(self,
                                  trades: OrderedDict,
                                  price_history: pd.DataFrame) -> Tuple[go.layout.Annotation]:
        """Creates annotations of the new trades after the last one in the chart.

        Parameters
        ----------
        trades : `OrderedDict`
            The history of trades for the current episode.
        price_history : `pd.DataFrame`
            The price history of the current episode.

        Returns
        -------
        `Tuple[go.layout.Annotation]`
            A tuple of annotations used in the renderering process.
        """
        annotations = []
        for trade in reversed(trades.values()):
            trade = trade[0]

            tp = float(trade.price)
            ts = float(trade.size)

            if trade.step <= self._last_trade_step:
                break

            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 15
                qty = round(ts / tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=qty,
                    size=ts,
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )

            elif trade.side.value == 'sell':
                color = 'FireBrick'
                ay = -15
                # qty = round(ts * tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=ts,
                    size=round(ts * tp, trade.base_instrument.precision),
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )
            else:
                raise ValueError(f"Valid trade side values are 'buy' and 'sell'. Found '{trade.side.value}'.")

            hovertext = 'Step {step} [{datetime}]<br>' \
                        '{side} {qty} {quote_instrument} @ {price} {base_instrument} {type}<br>' \
                        'Total: {size} {base_instrument} - Comm.: {commission}'.format(**text_info)

            annotations += [go.layout.Annotation(
                x=trade.step - 1, y=tp,
                ax=0, ay=ay, xref='x1', yref='y1', showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                hoverlabel=dict(bgcolor=color)
            )]

        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step

        return tuple(annotations)

    def render(self,
               episode: int,
               max_episodes: int,
               step: int,
               max_steps: int,
               price_history: pd.DataFrame,
               net_worth: pd.Series,
               performance: pd.DataFrame,
               trades: OrderedDict) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure(performance.keys())

        if self._show_chart:  # ensure chart visibility through notebook cell reruns
            display(self.fig)

        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._price_chart.update(dict(
            open=price_history['open'],
            high=price_history['high'],
            low=price_history['low'],
            close=price_history['close']
        ))
        self.fig.layout.annotations += self._create_trade_annotations(trades, price_history)

        self._volume_chart.update({'y': price_history['volume']})

        for trace in self.fig.select_traces(row=3):
            trace.update({'y': performance[trace.name]})

        self._net_worth_chart.update({'y': net_worth})

        if self._show_chart:
            self.fig.show()

        if self._save_format:
            self.save()

    def save(self) -> None:
        """Saves the current chart to a file.

        Notes
        -----
        All formats other than HTML require Orca installed and server running.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
            check_valid_format(valid_formats, self._save_format)

        check_path(self._path)

        filename = create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self._last_trade_step = 0
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)

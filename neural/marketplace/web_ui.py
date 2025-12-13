"""
Marketplace Web UI - Interactive web interface for browsing and managing models.
"""

from __future__ import annotations

from typing import Optional


try:
    import dash
    from dash import Dash, Input, Output, State, callback_context, dcc, html
    import dash_bootstrap_components as dbc
    from dash_bootstrap_components import themes
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .huggingface_integration import HuggingFaceIntegration
from .registry import ModelRegistry
from .search import SemanticSearch


class MarketplaceUI:
    """Web UI for Neural Marketplace."""

    def __init__(
        self,
        registry_dir: str = "neural_marketplace_registry",
        hf_token: Optional[str] = None
    ):
        """Initialize marketplace UI.

        Parameters
        ----------
        registry_dir : str
            Registry directory
        hf_token : str, optional
            HuggingFace API token
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash is not installed. "
                "Install it with: pip install dash dash-bootstrap-components"
            )

        self.registry = ModelRegistry(registry_dir)
        self.search = SemanticSearch(self.registry)

        try:
            self.hf = HuggingFaceIntegration(hf_token)
            self.hf_available = True
        except ImportError:
            self.hf = None
            self.hf_available = False

        self.app = Dash(
            __name__,
            external_stylesheets=[themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True
        )

        self.app.layout = self._create_layout()
        self._setup_callbacks()

    def _create_layout(self):
        """Create the main layout."""
        return html.Div([
            # Header
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.I(className="fas fa-store me-2"),
                                html.Span("Neural Marketplace", className="fs-4 fw-bold")
                            ]),
                            width="auto"
                        ),
                    ], align="center"),
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Browse", href="#", id="nav-browse", active=True)),
                        dbc.NavItem(dbc.NavLink("Upload", href="#", id="nav-upload")),
                        dbc.NavItem(dbc.NavLink("My Models", href="#", id="nav-my-models")),
                        dbc.NavItem(dbc.NavLink("HuggingFace", href="#", id="nav-hf")) if self.hf_available else None,
                    ], pills=True),
                ], fluid=True),
                color="primary",
                dark=True,
                className="mb-4"
            ),

            # Main content
            dbc.Container([
                # Store for current view
                dcc.Store(id='current-view', data='browse'),

                # Content area
                html.Div(id='content-area')
            ], fluid=True)
        ])

    def _create_browse_view(self):
        """Create browse/search view."""
        return html.Div([
            # Search bar
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            placeholder="Search models by name, description, tags...",
                            type="text"
                        ),
                        dbc.Button(
                            html.I(className="fas fa-search"),
                            id="search-button",
                            color="primary"
                        )
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Select(
                        id="sort-select",
                        options=[
                            {"label": "Most Recent", "value": "uploaded_at"},
                            {"label": "Most Popular", "value": "downloads"},
                            {"label": "Name (A-Z)", "value": "name"}
                        ],
                        value="uploaded_at"
                    )
                ], width=2),
                dbc.Col([
                    dbc.Button(
                        "Filters",
                        id="filter-button",
                        color="secondary",
                        outline=True
                    )
                ], width=2)
            ], className="mb-4"),

            # Filter panel (collapsible)
            dbc.Collapse(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Author"),
                                dbc.Input(id="filter-author", placeholder="Author name")
                            ], width=4),
                            dbc.Col([
                                html.Label("License"),
                                dbc.Select(
                                    id="filter-license",
                                    options=[
                                        {"label": "Any", "value": ""},
                                        {"label": "MIT", "value": "MIT"},
                                        {"label": "Apache-2.0", "value": "Apache-2.0"},
                                        {"label": "GPL-3.0", "value": "GPL-3.0"},
                                    ]
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Tags (comma-separated)"),
                                dbc.Input(id="filter-tags", placeholder="tag1, tag2")
                            ], width=4)
                        ])
                    ])
                ], className="mb-4"),
                id="filter-collapse",
                is_open=False
            ),

            # Statistics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="stat-total-models", className="text-primary"),
                            html.P("Total Models", className="text-muted mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="stat-total-downloads", className="text-success"),
                            html.P("Total Downloads", className="text-muted mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="stat-total-authors", className="text-info"),
                            html.P("Authors", className="text-muted mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="stat-total-tags", className="text-warning"),
                            html.P("Tags", className="text-muted mb-0")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),

            # Trending tags
            html.Div([
                html.H5("Trending Tags", className="mb-2"),
                html.Div(id="trending-tags", className="mb-4")
            ]),

            # Model list
            html.Div(id="model-list"),

            # Model details modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
                dbc.ModalBody(id="modal-body"),
                dbc.ModalFooter([
                    dbc.Button("Download", id="modal-download-btn", color="primary"),
                    dbc.Button("Close", id="modal-close-btn", color="secondary")
                ])
            ], id="model-modal", size="lg", is_open=False)
        ])

    def _create_upload_view(self):
        """Create model upload view."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Upload Model")),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Model Name*"),
                                        dbc.Input(
                                            id="upload-name",
                                            placeholder="My Awesome Model",
                                            required=True
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Author*"),
                                        dbc.Input(
                                            id="upload-author",
                                            placeholder="Your Name",
                                            required=True
                                        )
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Model File Path*"),
                                        dbc.Input(
                                            id="upload-path",
                                            placeholder="/path/to/model.neural",
                                            required=True
                                        )
                                    ], width=8),
                                    dbc.Col([
                                        dbc.Label("Version"),
                                        dbc.Input(
                                            id="upload-version",
                                            placeholder="1.0.0",
                                            value="1.0.0"
                                        )
                                    ], width=4)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Description"),
                                        dbc.Textarea(
                                            id="upload-description",
                                            placeholder="Describe your model...",
                                            rows=3
                                        )
                                    ])
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("License"),
                                        dbc.Select(
                                            id="upload-license",
                                            options=[
                                                {"label": "MIT", "value": "MIT"},
                                                {"label": "Apache-2.0", "value": "Apache-2.0"},
                                                {"label": "GPL-3.0", "value": "GPL-3.0"},
                                                {"label": "BSD-3-Clause", "value": "BSD-3-Clause"},
                                            ],
                                            value="MIT"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Label("Framework"),
                                        dbc.Input(
                                            id="upload-framework",
                                            placeholder="neural-dsl",
                                            value="neural-dsl"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Label("Tags (comma-separated)"),
                                        dbc.Input(
                                            id="upload-tags",
                                            placeholder="classification, cnn, resnet"
                                        )
                                    ], width=4)
                                ], className="mb-3"),
                                dbc.Button(
                                    "Upload Model",
                                    id="upload-submit-btn",
                                    color="primary",
                                    size="lg",
                                    className="w-100"
                                )
                            ])
                        ])
                    ])
                ], width=8)
            ], justify="center"),

            # Upload result
            html.Div(id="upload-result", className="mt-3")
        ])

    def _create_my_models_view(self):
        """Create my models view."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("My Models"),
                    dbc.Input(
                        id="my-models-author",
                        placeholder="Enter your author name",
                        className="mb-3"
                    ),
                    html.Div(id="my-models-list")
                ])
            ])
        ])

    def _create_hf_view(self):
        """Create HuggingFace Hub integration view."""
        return html.Div([
            dbc.Tabs([
                dbc.Tab(label="Search Hub", tab_id="hf-search"),
                dbc.Tab(label="Upload to Hub", tab_id="hf-upload"),
            ], id="hf-tabs", active_tab="hf-search"),
            html.Div(id="hf-content", className="mt-3")
        ])

    def _setup_callbacks(self):
        """Setup Dash callbacks."""

        @self.app.callback(
            Output('current-view', 'data'),
            [Input('nav-browse', 'n_clicks'),
             Input('nav-upload', 'n_clicks'),
             Input('nav-my-models', 'n_clicks'),
             Input('nav-hf', 'n_clicks')]
        )
        def update_current_view(browse, upload, my_models, hf):
            ctx = callback_context
            if not ctx.triggered:
                return 'browse'

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'nav-browse':
                return 'browse'
            elif button_id == 'nav-upload':
                return 'upload'
            elif button_id == 'nav-my-models':
                return 'my-models'
            elif button_id == 'nav-hf':
                return 'hf'

            return 'browse'

        @self.app.callback(
            Output('content-area', 'children'),
            Input('current-view', 'data')
        )
        def render_content(view):
            if view == 'browse':
                return self._create_browse_view()
            elif view == 'upload':
                return self._create_upload_view()
            elif view == 'my-models':
                return self._create_my_models_view()
            elif view == 'hf':
                return self._create_hf_view()
            return html.Div("Not found")

        @self.app.callback(
            Output('filter-collapse', 'is_open'),
            Input('filter-button', 'n_clicks'),
            State('filter-collapse', 'is_open')
        )
        def toggle_filters(n, is_open):
            if n:
                return not is_open
            return is_open

        @self.app.callback(
            [Output('stat-total-models', 'children'),
             Output('stat-total-downloads', 'children'),
             Output('stat-total-authors', 'children'),
             Output('stat-total-tags', 'children')],
            Input('current-view', 'data')
        )
        def update_stats(view):
            if view != 'browse':
                return dash.no_update

            total_models = len(self.registry.metadata["models"])
            total_downloads = sum(
                s.get("downloads", 0)
                for s in self.registry.stats.values()
            )
            total_authors = len(self.registry.metadata["authors"])
            total_tags = len(self.registry.metadata["tags"])

            return (
                f"{total_models:,}",
                f"{total_downloads:,}",
                f"{total_authors:,}",
                f"{total_tags:,}"
            )

        @self.app.callback(
            Output('trending-tags', 'children'),
            Input('current-view', 'data')
        )
        def update_trending_tags(view):
            if view != 'browse':
                return dash.no_update

            tags = self.search.get_trending_tags(limit=15)

            tag_badges = []
            for tag, count in tags:
                tag_badges.append(
                    dbc.Badge(
                        f"{tag} ({count})",
                        color="light",
                        text_color="dark",
                        className="me-2 mb-2",
                        pill=True
                    )
                )

            return tag_badges

        @self.app.callback(
            Output('model-list', 'children'),
            [Input('search-button', 'n_clicks'),
             Input('sort-select', 'value'),
             Input('current-view', 'data')],
            [State('search-input', 'value'),
             State('filter-author', 'value'),
             State('filter-license', 'value'),
             State('filter-tags', 'value')]
        )
        def update_model_list(n_clicks, sort_by, view, query, author, license_filter, tags):
            if view != 'browse':
                return dash.no_update

            # Search or list models
            if query and query.strip():
                filters = {}
                if author:
                    filters['author'] = author
                if license_filter:
                    filters['license'] = license_filter
                if tags:
                    filters['tags'] = [t.strip() for t in tags.split(',')]

                results = self.search.search(query, limit=50, filters=filters)
                models = [r[2] for r in results]
            else:
                models = self.registry.list_models(
                    author=author if author else None,
                    tags=[t.strip() for t in tags.split(',')] if tags else None,
                    sort_by=sort_by,
                    limit=50
                )

                # Apply license filter manually if needed
                if license_filter:
                    models = [m for m in models if m.get('license') == license_filter]

            # Create model cards
            cards = []
            for model in models:
                card = dbc.Card([
                    dbc.CardBody([
                        html.H5(model['name'], className="card-title"),
                        html.P(
                            f"by {model['author']}",
                            className="text-muted small mb-2"
                        ),
                        html.P(
                            model['description'][:100] + "..." if len(model['description']) > 100 else model['description'],
                            className="card-text"
                        ),
                        html.Div([
                            dbc.Badge(tag, color="info", className="me-1")
                            for tag in model.get('tags', [])[:5]
                        ], className="mb-2"),
                        html.Hr(),
                        html.Div([
                            html.Span([
                                html.I(className="fas fa-download me-1"),
                                f"{model.get('downloads', 0)}"
                            ], className="me-3"),
                            html.Span([
                                html.I(className="fas fa-code-branch me-1"),
                                model.get('version', '1.0.0')
                            ], className="me-3"),
                            html.Span([
                                html.I(className="fas fa-balance-scale me-1"),
                                model.get('license', 'MIT')
                            ])
                        ], className="small text-muted"),
                        dbc.Button(
                            "View Details",
                            id={'type': 'view-model-btn', 'index': model['id']},
                            color="primary",
                            size="sm",
                            className="mt-2"
                        )
                    ])
                ], className="mb-3")
                cards.append(dbc.Col(card, width=12, lg=6, xl=4))

            if not cards:
                return html.Div(
                    dbc.Alert("No models found", color="info"),
                    className="text-center"
                )

            return dbc.Row(cards)

        @self.app.callback(
            Output('upload-result', 'children'),
            Input('upload-submit-btn', 'n_clicks'),
            [State('upload-name', 'value'),
             State('upload-author', 'value'),
             State('upload-path', 'value'),
             State('upload-version', 'value'),
             State('upload-description', 'value'),
             State('upload-license', 'value'),
             State('upload-framework', 'value'),
             State('upload-tags', 'value')]
        )
        def handle_upload(n_clicks, name, author, path, version, description, license, framework, tags):
            if not n_clicks:
                return dash.no_update

            if not all([name, author, path]):
                return dbc.Alert("Please fill in all required fields", color="danger")

            try:
                tags_list = [t.strip() for t in tags.split(',')] if tags else []

                model_id = self.registry.upload_model(
                    name=name,
                    author=author,
                    model_path=path,
                    description=description or "",
                    license=license,
                    tags=tags_list,
                    framework=framework,
                    version=version or "1.0.0"
                )

                return dbc.Alert(
                    [
                        html.H4("Success!", className="alert-heading"),
                        html.P("Model uploaded successfully!"),
                        html.Hr(),
                        html.P(f"Model ID: {model_id}", className="mb-0")
                    ],
                    color="success"
                )
            except Exception as e:
                return dbc.Alert(f"Upload failed: {str(e)}", color="danger")

        @self.app.callback(
            Output('my-models-list', 'children'),
            Input('my-models-author', 'value')
        )
        def update_my_models(author):
            if not author or not author.strip():
                return html.Div("Enter your author name to view your models")

            models = self.registry.list_models(author=author)

            if not models:
                return dbc.Alert(f"No models found for author: {author}", color="info")

            table_header = [
                html.Thead(html.Tr([
                    html.Th("Name"),
                    html.Th("Version"),
                    html.Th("Downloads"),
                    html.Th("Uploaded"),
                    html.Th("Actions")
                ]))
            ]

            rows = []
            for model in models:
                row = html.Tr([
                    html.Td(model['name']),
                    html.Td(model.get('version', '1.0.0')),
                    html.Td(model.get('downloads', 0)),
                    html.Td(model['uploaded_at'][:10]),
                    html.Td([
                        dbc.Button(
                            "Edit",
                            id={'type': 'edit-model-btn', 'index': model['id']},
                            color="warning",
                            size="sm",
                            className="me-2"
                        ),
                        dbc.Button(
                            "Delete",
                            id={'type': 'delete-model-btn', 'index': model['id']},
                            color="danger",
                            size="sm"
                        )
                    ])
                ])
                rows.append(row)

            table_body = [html.Tbody(rows)]

            return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True)

    def run(self, host: str = '0.0.0.0', port: int = 8052, debug: bool = False):
        """Run the web UI.

        Parameters
        ----------
        host : str
            Host address
        port : int
            Port number
        debug : bool
            Debug mode
        """
        self.app.run_server(host=host, port=port, debug=debug)

    def get_app(self):
        """Get Dash app instance.

        Returns
        -------
        Dash
            Dash app
        """
        return self.app

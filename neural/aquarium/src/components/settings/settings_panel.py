from __future__ import annotations

from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from .config_manager import ConfigManager


class SettingsPanel:
    """Settings and preferences panel for Aquarium IDE"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def create_layout(self) -> html.Div:
        """Create the settings panel layout"""
        return html.Div([
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Settings & Preferences")),
                dbc.ModalBody([
                    dbc.Tabs([
                        self._create_editor_tab(),
                        self._create_keybindings_tab(),
                        self._create_python_tab(),
                        self._create_backend_tab(),
                        self._create_autosave_tab(),
                        self._create_extensions_tab(),
                        self._create_advanced_tab()
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Reset to Defaults", id="settings-reset-btn", color="warning", className="me-2"),
                    dbc.Button("Export Settings", id="settings-export-btn", color="info", className="me-2"),
                    dbc.Button("Import Settings", id="settings-import-btn", color="info", className="me-2"),
                    dbc.Button("Cancel", id="settings-cancel-btn", color="secondary", className="me-2"),
                    dbc.Button("Save", id="settings-save-btn", color="primary")
                ])
            ], id="settings-modal", size="xl", scrollable=True),
            
            dcc.Store(id="settings-store", data=self.config_manager.get_all()),
            html.Div(id="settings-notification")
        ])
    
    def _create_editor_tab(self) -> dbc.Tab:
        """Create editor settings tab"""
        config = self.config_manager.get('editor')
        
        return dbc.Tab(label="Editor", children=[
            html.Div([
                html.H5("Appearance", className="mt-3 mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Theme"),
                        dcc.Dropdown(
                            id="editor-theme",
                            options=[
                                {'label': 'Light', 'value': 'light'},
                                {'label': 'Dark', 'value': 'dark'},
                                {'label': 'Custom', 'value': 'custom'}
                            ],
                            value=config.get('theme', 'dark'),
                            clearable=False
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Font Size"),
                        dbc.Input(
                            id="editor-font-size",
                            type="number",
                            value=config.get('font_size', 14),
                            min=8,
                            max=32
                        )
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Font Family"),
                        dbc.Input(
                            id="editor-font-family",
                            type="text",
                            value=config.get('font_family', 'Consolas, Monaco, monospace')
                        )
                    ], width=12)
                ], className="mb-3"),
                
                html.Div(id="custom-theme-container", children=[
                    html.H6("Custom Theme Colors", className="mt-3 mb-2"),
                    self._create_custom_theme_inputs(config.get('custom_theme', {}))
                ], style={'display': 'block' if config.get('theme') == 'custom' else 'none'}),
                
                html.H5("Behavior", className="mt-4 mb-3"),
                
                dbc.Checklist(
                    id="editor-behavior",
                    options=[
                        {'label': 'Show Line Numbers', 'value': 'line_numbers'},
                        {'label': 'Word Wrap', 'value': 'word_wrap'},
                        {'label': 'Auto Indent', 'value': 'auto_indent'},
                        {'label': 'Bracket Matching', 'value': 'bracket_matching'},
                        {'label': 'Highlight Active Line', 'value': 'highlight_active_line'}
                    ],
                    value=[k for k, v in config.items() if k in ['line_numbers', 'word_wrap', 'auto_indent', 'bracket_matching', 'highlight_active_line'] and v],
                    switch=True
                ),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Tab Size", className="mt-3"),
                        dbc.Input(
                            id="editor-tab-size",
                            type="number",
                            value=config.get('tab_size', 4),
                            min=2,
                            max=8
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Insert Spaces", className="mt-3"),
                        dbc.Checklist(
                            id="editor-insert-spaces",
                            options=[{'label': 'Use spaces instead of tabs', 'value': True}],
                            value=[True] if config.get('insert_spaces', True) else [],
                            switch=True
                        )
                    ], width=6)
                ])
            ], className="p-3")
        ])
    
    def _create_custom_theme_inputs(self, custom_theme: dict) -> html.Div:
        """Create color picker inputs for custom theme"""
        colors = [
            ('background', 'Background', custom_theme.get('background', '#1e1e1e')),
            ('foreground', 'Foreground', custom_theme.get('foreground', '#d4d4d4')),
            ('selection', 'Selection', custom_theme.get('selection', '#264f78')),
            ('comment', 'Comment', custom_theme.get('comment', '#6a9955')),
            ('keyword', 'Keyword', custom_theme.get('keyword', '#569cd6')),
            ('string', 'String', custom_theme.get('string', '#ce9178')),
            ('number', 'Number', custom_theme.get('number', '#b5cea8')),
            ('function', 'Function', custom_theme.get('function', '#dcdcaa')),
            ('operator', 'Operator', custom_theme.get('operator', '#d4d4d4')),
            ('cursor', 'Cursor', custom_theme.get('cursor', '#ffffff'))
        ]
        
        rows = []
        for i in range(0, len(colors), 2):
            cols = []
            for j in range(2):
                if i + j < len(colors):
                    key, label, value = colors[i + j]
                    cols.append(dbc.Col([
                        dbc.Label(label),
                        dbc.Input(
                            id=f"theme-color-{key}",
                            type="color",
                            value=value,
                            className="form-control-color"
                        )
                    ], width=6))
            rows.append(dbc.Row(cols, className="mb-2"))
        
        return html.Div(rows)
    
    def _create_keybindings_tab(self) -> dbc.Tab:
        """Create keybindings settings tab"""
        config = self.config_manager.get('keybindings')
        
        keybinding_groups = {
            'File Operations': ['save', 'save_as', 'open', 'new_file', 'close_tab'],
            'Search & Navigate': ['find', 'replace', 'goto_line', 'command_palette'],
            'Editing': ['comment_line', 'indent', 'outdent', 'duplicate_line', 'delete_line', 'move_line_up', 'move_line_down'],
            'View': ['toggle_terminal', 'toggle_sidebar'],
            'Model Operations': ['run_model', 'debug_model', 'compile_model']
        }
        
        children = []
        for group_name, keys in keybinding_groups.items():
            children.append(html.H6(group_name, className="mt-3 mb-2"))
            for key in keys:
                label = key.replace('_', ' ').title()
                children.append(
                    dbc.Row([
                        dbc.Col(dbc.Label(label), width=4),
                        dbc.Col(
                            dbc.Input(
                                id=f"keybinding-{key}",
                                type="text",
                                value=config.get(key, ''),
                                placeholder="e.g., Ctrl+S"
                            ),
                            width=8
                        )
                    ], className="mb-2")
                )
        
        return dbc.Tab(label="Keybindings", children=[
            html.Div([
                html.H5("Keyboard Shortcuts", className="mt-3 mb-3"),
                html.P("Customize keyboard shortcuts for common actions. Use modifiers: Ctrl, Shift, Alt, and keys.", 
                       className="text-muted"),
                html.Div(children, className="keybindings-list")
            ], className="p-3")
        ])
    
    def _create_python_tab(self) -> dbc.Tab:
        """Create Python interpreter settings tab"""
        config = self.config_manager.get('python')
        
        return dbc.Tab(label="Python", children=[
            html.Div([
                html.H5("Python Interpreter", className="mt-3 mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Interpreter Path"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="python-interpreter-path",
                                type="text",
                                value=config.get('interpreter_path', ''),
                                placeholder="Path to Python executable"
                            ),
                            dbc.Button("Browse", id="python-browse-btn", color="secondary")
                        ])
                    ], width=12)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Default Interpreter"),
                        dbc.Input(
                            id="python-default-interpreter",
                            type="text",
                            value=config.get('default_interpreter', 'python'),
                            placeholder="python, python3, etc."
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Checklist(
                            id="python-use-system",
                            options=[{'label': 'Use System Python', 'value': True}],
                            value=[True] if config.get('use_system_python', True) else [],
                            switch=True,
                            className="mt-4"
                        )
                    ], width=6)
                ], className="mb-3"),
                
                html.H6("Virtual Environment", className="mt-3 mb-2"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Virtual Environment Path"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="python-venv-path",
                                type="text",
                                value=config.get('virtual_env_path', ''),
                                placeholder="Path to virtual environment"
                            ),
                            dbc.Button("Browse", id="python-venv-browse-btn", color="secondary")
                        ])
                    ], width=12)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Conda Environment"),
                        dbc.Input(
                            id="python-conda-env",
                            type="text",
                            value=config.get('conda_env', ''),
                            placeholder="Conda environment name"
                        )
                    ], width=12)
                ], className="mb-3"),
                
                html.Div([
                    dbc.Button("Detect Interpreters", id="python-detect-btn", color="info", className="me-2"),
                    dbc.Button("Test Interpreter", id="python-test-btn", color="success")
                ]),
                
                html.Div(id="python-test-output", className="mt-3")
            ], className="p-3")
        ])
    
    def _create_backend_tab(self) -> dbc.Tab:
        """Create backend configuration settings tab"""
        config = self.config_manager.get('backend')
        preferences = config.get('preferences', {})
        
        return dbc.Tab(label="Backend", children=[
            html.Div([
                html.H5("Default Backend", className="mt-3 mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Preferred Backend"),
                        dcc.Dropdown(
                            id="backend-default",
                            options=[
                                {'label': 'TensorFlow', 'value': 'tensorflow'},
                                {'label': 'PyTorch', 'value': 'pytorch'},
                                {'label': 'ONNX', 'value': 'onnx'}
                            ],
                            value=config.get('default', 'tensorflow'),
                            clearable=False
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Checklist(
                            id="backend-auto-detect",
                            options=[{'label': 'Auto-detect available backends', 'value': True}],
                            value=[True] if config.get('auto_detect', True) else [],
                            switch=True,
                            className="mt-4"
                        )
                    ], width=6)
                ], className="mb-3"),
                
                html.H6("Backend Preferences", className="mt-3 mb-2"),
                html.P("Enable or disable specific backends", className="text-muted"),
                
                dbc.Checklist(
                    id="backend-preferences",
                    options=[
                        {'label': 'TensorFlow', 'value': 'tensorflow'},
                        {'label': 'PyTorch', 'value': 'pytorch'},
                        {'label': 'ONNX', 'value': 'onnx'}
                    ],
                    value=[k for k, v in preferences.items() if v],
                    switch=True
                ),
                
                html.H6("Backend Status", className="mt-4 mb-2"),
                html.Div(id="backend-status", children=[
                    html.P("Click 'Check Backends' to detect installed backends", className="text-muted")
                ]),
                
                dbc.Button("Check Backends", id="backend-check-btn", color="info", className="mt-2")
            ], className="p-3")
        ])
    
    def _create_autosave_tab(self) -> dbc.Tab:
        """Create autosave settings tab"""
        config = self.config_manager.get('autosave')
        
        return dbc.Tab(label="Auto-save", children=[
            html.Div([
                html.H5("Auto-save Settings", className="mt-3 mb-3"),
                
                dbc.Checklist(
                    id="autosave-enabled",
                    options=[{'label': 'Enable Auto-save', 'value': True}],
                    value=[True] if config.get('enabled', True) else [],
                    switch=True,
                    className="mb-3"
                ),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Auto-save Interval (seconds)"),
                        dbc.Input(
                            id="autosave-interval",
                            type="number",
                            value=config.get('interval', 30),
                            min=5,
                            max=300,
                            step=5
                        )
                    ], width=6)
                ], className="mb-3"),
                
                html.H6("Additional Options", className="mt-3 mb-2"),
                
                dbc.Checklist(
                    id="autosave-options",
                    options=[
                        {'label': 'Auto-save on focus lost', 'value': 'on_focus_lost'},
                        {'label': 'Auto-save on window change', 'value': 'on_window_change'}
                    ],
                    value=[k for k, v in config.items() if k in ['on_focus_lost', 'on_window_change'] and v],
                    switch=True
                ),
                
                html.Div([
                    html.Hr(),
                    html.P([
                        html.I(className="fas fa-info-circle me-2"),
                        "Auto-save helps prevent data loss by automatically saving your work at regular intervals."
                    ], className="text-muted small")
                ], className="mt-4")
            ], className="p-3")
        ])
    
    def _create_extensions_tab(self) -> dbc.Tab:
        """Create extensions and plugins settings tab"""
        ext_config = self.config_manager.get('extensions')
        plugin_config = self.config_manager.get('plugins')
        
        return dbc.Tab(label="Extensions", children=[
            html.Div([
                html.H5("Extensions & Plugins", className="mt-3 mb-3"),
                
                dbc.Checklist(
                    id="extensions-auto-update",
                    options=[
                        {'label': 'Auto-update extensions', 'value': 'auto_update'},
                        {'label': 'Check for updates on startup', 'value': 'check_updates'}
                    ],
                    value=[
                        k for k in ['auto_update', 'check_updates'] 
                        if (k == 'auto_update' and ext_config.get('auto_update', True)) or 
                           (k == 'check_updates' and ext_config.get('check_updates_on_startup', True))
                    ],
                    switch=True,
                    className="mb-3"
                ),
                
                html.H6("Installed Extensions", className="mt-3 mb-2"),
                html.Div(id="extensions-list", children=[
                    self._create_extension_list(ext_config.get('enabled', []), ext_config.get('disabled', []))
                ]),
                
                dbc.Button("Browse Extension Marketplace", id="extensions-browse-btn", color="primary", className="mt-2 me-2"),
                dbc.Button("Install from File", id="extensions-install-btn", color="secondary", className="mt-2"),
                
                html.Hr(className="my-4"),
                
                html.H6("Plugin Configuration", className="mt-3 mb-2"),
                
                dbc.Checklist(
                    id="plugins-auto-install",
                    options=[{'label': 'Auto-install plugin dependencies', 'value': True}],
                    value=[True] if plugin_config.get('auto_install_dependencies', True) else [],
                    switch=True
                ),
                
                html.Div(id="plugins-list", children=[
                    self._create_plugin_list(plugin_config.get('installed', []))
                ], className="mt-3")
            ], className="p-3")
        ])
    
    def _create_extension_list(self, enabled: list, disabled: list) -> html.Div:
        """Create list of installed extensions"""
        if not enabled and not disabled:
            return html.P("No extensions installed", className="text-muted")
        
        items = []
        for ext in enabled:
            items.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Span(ext, className="fw-bold"), width=6),
                            dbc.Col([
                                dbc.Badge("Enabled", color="success", className="me-2"),
                                dbc.Button("Disable", id={"type": "ext-disable", "id": ext}, size="sm", color="warning", className="me-1"),
                                dbc.Button("Remove", id={"type": "ext-remove", "id": ext}, size="sm", color="danger")
                            ], width=6, className="text-end")
                        ])
                    ])
                ], className="mb-2")
            )
        
        for ext in disabled:
            items.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Span(ext, className="fw-bold text-muted"), width=6),
                            dbc.Col([
                                dbc.Badge("Disabled", color="secondary", className="me-2"),
                                dbc.Button("Enable", id={"type": "ext-enable", "id": ext}, size="sm", color="success", className="me-1"),
                                dbc.Button("Remove", id={"type": "ext-remove", "id": ext}, size="sm", color="danger")
                            ], width=6, className="text-end")
                        ])
                    ])
                ], className="mb-2")
            )
        
        return html.Div(items)
    
    def _create_plugin_list(self, installed: list) -> html.Div:
        """Create list of installed plugins"""
        if not installed:
            return html.P("No plugins installed", className="text-muted")
        
        items = []
        for plugin in installed:
            items.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Span(plugin.get('name', 'Unknown'), className="fw-bold"),
                                html.Br(),
                                html.Small(plugin.get('version', '0.0.0'), className="text-muted")
                            ], width=6),
                            dbc.Col([
                                dbc.Button("Configure", id={"type": "plugin-config", "id": plugin.get('id', '')}, size="sm", color="info", className="me-1"),
                                dbc.Button("Remove", id={"type": "plugin-remove", "id": plugin.get('id', '')}, size="sm", color="danger")
                            ], width=6, className="text-end")
                        ])
                    ])
                ], className="mb-2")
            )
        
        return html.Div(items)
    
    def _create_advanced_tab(self) -> dbc.Tab:
        """Create advanced settings tab"""
        ui_config = self.config_manager.get('ui')
        terminal_config = self.config_manager.get('terminal')
        neuraldbg_config = self.config_manager.get('neuraldbg')
        
        return dbc.Tab(label="Advanced", children=[
            html.Div([
                html.H5("User Interface", className="mt-3 mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Sidebar Width (px)"),
                        dbc.Input(
                            id="ui-sidebar-width",
                            type="number",
                            value=ui_config.get('sidebar_width', 250),
                            min=150,
                            max=500
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Panel Height (px)"),
                        dbc.Input(
                            id="ui-panel-height",
                            type="number",
                            value=ui_config.get('panel_height', 200),
                            min=100,
                            max=500
                        )
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Checklist(
                    id="ui-features",
                    options=[
                        {'label': 'Show Minimap', 'value': 'show_minimap'},
                        {'label': 'Show Breadcrumbs', 'value': 'show_breadcrumbs'},
                        {'label': 'Show Status Bar', 'value': 'show_status_bar'},
                        {'label': 'Show Activity Bar', 'value': 'show_activity_bar'}
                    ],
                    value=[k for k, v in ui_config.items() if k.startswith('show_') and v],
                    switch=True
                ),
                
                html.Hr(className="my-4"),
                
                html.H5("Terminal", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Default Shell"),
                        dbc.Input(
                            id="terminal-shell",
                            type="text",
                            value=terminal_config.get('shell', 'powershell')
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Font Size"),
                        dbc.Input(
                            id="terminal-font-size",
                            type="number",
                            value=terminal_config.get('font_size', 12),
                            min=8,
                            max=24
                        )
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Cursor Style"),
                        dcc.Dropdown(
                            id="terminal-cursor-style",
                            options=[
                                {'label': 'Block', 'value': 'block'},
                                {'label': 'Underline', 'value': 'underline'},
                                {'label': 'Bar', 'value': 'bar'}
                            ],
                            value=terminal_config.get('cursor_style', 'block'),
                            clearable=False
                        )
                    ], width=6)
                ], className="mb-3"),
                
                html.Hr(className="my-4"),
                
                html.H5("NeuralDbg Integration", className="mb-3"),
                
                dbc.Checklist(
                    id="neuraldbg-auto-launch",
                    options=[{'label': 'Auto-launch NeuralDbg on debug', 'value': True}],
                    value=[True] if neuraldbg_config.get('auto_launch', False) else [],
                    switch=True,
                    className="mb-3"
                ),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Port"),
                        dbc.Input(
                            id="neuraldbg-port",
                            type="number",
                            value=neuraldbg_config.get('port', 8050),
                            min=1024,
                            max=65535
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Host"),
                        dbc.Input(
                            id="neuraldbg-host",
                            type="text",
                            value=neuraldbg_config.get('host', 'localhost')
                        )
                    ], width=6)
                ], className="mb-3"),
                
                html.Hr(className="my-4"),
                
                html.Div([
                    dbc.Button("Clear Cache", id="advanced-clear-cache-btn", color="warning", className="me-2"),
                    dbc.Button("Reset All Settings", id="advanced-reset-all-btn", color="danger")
                ])
            ], className="p-3")
        ])
    
    def register_callbacks(self, app):
        """Register all callbacks for the settings panel"""
        
        @app.callback(
            Output("settings-modal", "is_open"),
            [Input("settings-open-btn", "n_clicks"),
             Input("settings-save-btn", "n_clicks"),
             Input("settings-cancel-btn", "n_clicks")],
            [State("settings-modal", "is_open")]
        )
        def toggle_settings_modal(open_clicks, save_clicks, cancel_clicks, is_open):
            from dash import callback_context
            if not callback_context.triggered:
                return False
            
            trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
            if trigger_id in ["settings-save-btn", "settings-cancel-btn"]:
                return False
            return not is_open
        
        @app.callback(
            Output("custom-theme-container", "style"),
            Input("editor-theme", "value")
        )
        def toggle_custom_theme_inputs(theme):
            if theme == "custom":
                return {'display': 'block'}
            return {'display': 'none'}
        
        @app.callback(
            Output("settings-store", "data"),
            Input("settings-save-btn", "n_clicks"),
            [State("editor-theme", "value"),
             State("editor-font-size", "value"),
             State("editor-font-family", "value"),
             State("editor-behavior", "value"),
             State("editor-tab-size", "value"),
             State("editor-insert-spaces", "value"),
             State("python-interpreter-path", "value"),
             State("python-default-interpreter", "value"),
             State("python-use-system", "value"),
             State("python-venv-path", "value"),
             State("python-conda-env", "value"),
             State("backend-default", "value"),
             State("backend-auto-detect", "value"),
             State("backend-preferences", "value"),
             State("autosave-enabled", "value"),
             State("autosave-interval", "value"),
             State("autosave-options", "value"),
             State("extensions-auto-update", "value"),
             State("plugins-auto-install", "value"),
             State("ui-sidebar-width", "value"),
             State("ui-panel-height", "value"),
             State("ui-features", "value"),
             State("terminal-shell", "value"),
             State("terminal-font-size", "value"),
             State("terminal-cursor-style", "value"),
             State("neuraldbg-auto-launch", "value"),
             State("neuraldbg-port", "value"),
             State("neuraldbg-host", "value")],
            prevent_initial_call=True
        )
        def save_settings(n_clicks, *args):
            if not n_clicks:
                return self.config_manager.get_all()
            
            # Update configuration with new values
            (editor_theme, editor_font_size, editor_font_family, editor_behavior, 
             editor_tab_size, editor_insert_spaces, python_interpreter_path, 
             python_default_interpreter, python_use_system, python_venv_path, 
             python_conda_env, backend_default, backend_auto_detect, backend_preferences,
             autosave_enabled, autosave_interval, autosave_options, extensions_auto_update,
             plugins_auto_install, ui_sidebar_width, ui_panel_height,
             ui_features, terminal_shell, terminal_font_size, terminal_cursor_style,
             neuraldbg_auto_launch, neuraldbg_port, neuraldbg_host) = args
            
            # Update editor settings
            self.config_manager.update_section('editor', {
                'theme': editor_theme,
                'font_size': editor_font_size,
                'font_family': editor_font_family,
                'line_numbers': 'line_numbers' in (editor_behavior or []),
                'word_wrap': 'word_wrap' in (editor_behavior or []),
                'auto_indent': 'auto_indent' in (editor_behavior or []),
                'bracket_matching': 'bracket_matching' in (editor_behavior or []),
                'highlight_active_line': 'highlight_active_line' in (editor_behavior or []),
                'tab_size': editor_tab_size,
                'insert_spaces': bool(editor_insert_spaces)
            })
            
            # Update Python settings
            self.config_manager.update_section('python', {
                'interpreter_path': python_interpreter_path,
                'default_interpreter': python_default_interpreter,
                'use_system_python': bool(python_use_system),
                'virtual_env_path': python_venv_path,
                'conda_env': python_conda_env
            })
            
            # Update backend settings
            self.config_manager.update_section('backend', {
                'default': backend_default,
                'auto_detect': bool(backend_auto_detect),
                'preferences': {
                    'tensorflow': 'tensorflow' in (backend_preferences or []),
                    'pytorch': 'pytorch' in (backend_preferences or []),
                    'onnx': 'onnx' in (backend_preferences or [])
                }
            })
            
            # Update autosave settings
            self.config_manager.update_section('autosave', {
                'enabled': bool(autosave_enabled),
                'interval': autosave_interval,
                'on_focus_lost': 'on_focus_lost' in (autosave_options or []),
                'on_window_change': 'on_window_change' in (autosave_options or [])
            })
            
            # Update extensions settings
            self.config_manager.update_section('extensions', {
                'auto_update': 'auto_update' in (extensions_auto_update or []),
                'check_updates_on_startup': 'check_updates' in (extensions_auto_update or [])
            })
            
            # Update plugins settings
            self.config_manager.update_section('plugins', {
                'auto_install_dependencies': bool(plugins_auto_install)
            })
            
            # Update UI settings
            self.config_manager.update_section('ui', {
                'sidebar_width': ui_sidebar_width,
                'panel_height': ui_panel_height,
                'show_minimap': 'show_minimap' in (ui_features or []),
                'show_breadcrumbs': 'show_breadcrumbs' in (ui_features or []),
                'show_status_bar': 'show_status_bar' in (ui_features or []),
                'show_activity_bar': 'show_activity_bar' in (ui_features or [])
            })
            
            # Update terminal settings
            self.config_manager.update_section('terminal', {
                'shell': terminal_shell,
                'font_size': terminal_font_size,
                'cursor_style': terminal_cursor_style
            })
            
            # Update NeuralDbg settings
            self.config_manager.update_section('neuraldbg', {
                'auto_launch': bool(neuraldbg_auto_launch),
                'port': neuraldbg_port,
                'host': neuraldbg_host
            })
            
            return self.config_manager.get_all()
        
        @app.callback(
            Output("settings-notification", "children"),
            Input("settings-reset-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def reset_settings(n_clicks):
            if n_clicks:
                self.config_manager.reset_to_defaults()
                return dbc.Alert("Settings reset to defaults", color="info", duration=3000)
            return ""

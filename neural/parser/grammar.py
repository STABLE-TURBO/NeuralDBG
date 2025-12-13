"""Neural DSL Grammar Definition.

This module contains the Lark grammar specification for the Neural DSL.
"""

NEURAL_DSL_GRAMMAR = r"""
    // Layer type tokens (case-insensitive)
    DENSE: "dense"i
    MAXPOOLING1D: "maxpooling1d"i
    MAXPOOLING2D: "maxpooling2d"i
    MAXPOOLING3D: "maxpooling3d"i
    CONV2D: "conv2d"i
    CONV1D: "conv1d"i
    CONV3D: "conv3d"i
    DROPOUT: "dropout"i
    EMBEDDING: "embedding"i
    FLATTEN: "flatten"i
    LSTM: "lstm"i
    GRU: "gru"i
    SIMPLE_RNN_DROPOUT_WRAPPER.2: "simplernndropoutwrapper"i
    SIMPLERNN: "simplernn"i
    OUTPUT: "output"i
    TRANSFORMER: "transformer"i
    TRANSFORMER_ENCODER: "transformerencoder"i
    TRANSFORMER_DECODER: "transformerdecoder"i
    CONV2DTRANSPOSE: "conv2dtranspose"i
    LSTMCELL: "lstmcell"i
    GRUCELL: "grucell"i
    BATCHNORMALIZATION: "batchnormalization"i
    GAUSSIANNOISE: "gaussiannoise"i
    LAYERNORMALIZATION: "layernormalization"i
    INSTANCENORMALIZATION: "instancenormalization"i
    GROUPNORMALIZATION: "groupnormalization"i
    ACTIVATION: "activation"i
    ADD: "add"i
    SUBSTRACT: "subtract"i
    MULTIPLY: "multiply"i
    AVERAGE: "average"i
    MAXIMUM: "maximum"i
    CONCATENATE: "concatenate"i
    DOT: "dot"i
    TIMEDISTRIBUTED: "timedistributed"i
    RESIDUALCONNECTION: "residualconnection"i
    GLOBALAVERAGEPOOLING2D: "globalaveragepooling2d"i
    GLOBALAVERAGEPOOLING1D: "globalaveragepooling1d"i
    MULTIHEADATTENTION: "multiheadattention"i

    // Layer type tokens (case-insensitive)
    LAYER_TYPE.2: "dense"i | "conv2d"i | "conv1d"i | "conv3d"i | "dropout"i | "embedding"i | "flatten"i | "lstm"i | "gru"i | "simplernndropoutwrapper"i | "simplernn"i | "output"i| "transformer"i | "transformerencoder"i | "transformerdecoder"i | "conv2dtranspose"i | "maxpooling2d"i | "maxpooling1d"i | "maxpooling3d"i | "batchnormalization"i | "gaussiannoise"i | "instancenormalization"i | "groupnormalization"i | "activation"i | "add"i | "subtract"i | "multiply"i | "average"i | "maximum"i | "concatenate"i | "dot"i | "timedistributed"i | "residualconnection"i | "globalaveragepooling2d"i | "globalaveragepooling1d"i | "multiheadattention"i

    // Basic tokens
    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: /"[^"]*"/ | /'[^']*'/
    INT: /[+-]?[0-9]+/
    FLOAT: /[+-]?[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?/ | /[+-]?[0-9]+[eE][+-]?[0-9]+/
    NUMBER: INT | FLOAT
    TRUE.2: "true"i
    FALSE.2: "false"i
    NONE.2: "none"i
    BOOL: TRUE | FALSE
    AT: "@"

    // Layer name patterns
    NAMED_LAYER: /[a-zA-Z_][a-zA-Z0-9_]*_layer/i

    // Comments (inline and multi-line)
    COMMENT: "//" /[^\n]*/
    MULTILINE_COMMENT: "/*" /(.|\n)*?/ "*/"

    // Top-level network definition
    network: "network" NAME "{" network_config "}"

    // Network configuration sections
    network_config: config_item*
    config_item: input_definition
               | layers_definition
               | optimizer_definition
               | loss_definition
               | metrics_definition
               | training_definition
               | hpo_definition
               | execution_config_definition

    // Input definition
    input_definition: "input" ":" (input_layer | named_inputs)
    input_layer: tuple_ | "[" tuple_ ("," tuple_)* "]"
    named_inputs: "{" (NAME ":" tuple_)+ "}"

    // Layers definition
    layers_definition: "layers" ":" layer_or_repeated+
    layer_or_repeated: (basic_layer | advanced_layer) multiplier?
    multiplier: "*" INT

    // Basic layer with optional device specification
    basic_layer: LAYER_TYPE "(" param_style1? ")" device_spec? sublayers?

    // Advanced layer types
    advanced_layer: branch_spec

    // Device specification
    device_spec: AT STRING

    // Sublayers (for nested architectures)
    sublayers: "{" layer_or_repeated* "}"

    // Branch specification
    branch_spec: NAME ":" "{" layer_or_repeated* "}"

    // Parameter styles
    param_style1: params
    params: param ("," param)*
    param: named_param | hpo_expr | value

    // Named parameters
    named_param: NAME ":" (hpo_expr | value | lr_schedule)
    named_params: named_param ("," named_param)*

    // Learning rate schedules
    lr_schedule: NAME "(" lr_schedule_args? ")"
    lr_schedule_args: lr_schedule_arg ("," lr_schedule_arg)*
    lr_schedule_arg: hpo_expr | value

    // HPO expressions
    hpo_expr: "HPO" "(" hpo_type "(" hpo_args ")" ")"
    hpo_type: "range" | "log_range" | "choice" | "categorical"
    hpo_args: hpo_arg ("," hpo_arg)*
    hpo_arg: value | named_hpo_arg
    named_hpo_arg: NAME "=" value

    // Values
    value: tuple_ | list_ | dict_ | number | string_value | bool_value | NONE
    tuple_: "(" [value ("," value)*] ")"
    list_: "[" [value ("," value)*] "]"
    dict_: "{" [dict_pair ("," dict_pair)*] "}"
    dict_pair: (NAME | string_value) ":" value
    number: INT | FLOAT
    string_value: STRING
    bool_value: BOOL

    // Optimizer definition
    optimizer_definition: "optimizer" ":" optimizer_spec
    optimizer_spec: optimizer_named | optimizer_string
    optimizer_named: NAME "{" named_params "}"
    optimizer_string: STRING

    // Loss definition
    loss_definition: "loss" ":" (NAME | STRING)

    // Metrics definition
    metrics_definition: "metrics" ":" "[" metric_list "]"
    metric_list: (NAME | STRING) ("," (NAME | STRING))*

    // Training definition
    training_definition: "training" ":" "{" training_params "}"
    training_params: training_param ("," training_param)*
    training_param: NAME ":" value

    // HPO definition
    hpo_definition: "hpo" ":" "{" hpo_params "}"
    hpo_params: hpo_param_def ("," hpo_param_def)*
    hpo_param_def: NAME ":" value

    // Execution config definition
    execution_config_definition: "execution" ":" "{" execution_params "}"
    execution_params: execution_param ("," execution_param)*
    execution_param: NAME ":" value

    // Whitespace and comments handling
    %import common.WS
    %ignore WS
    %ignore COMMENT
    %ignore MULTILINE_COMMENT
"""

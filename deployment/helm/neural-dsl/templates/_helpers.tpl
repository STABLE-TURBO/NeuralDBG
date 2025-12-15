{{/*
Expand the name of the chart.
*/}}
{{- define "neural-dsl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "neural-dsl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "neural-dsl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "neural-dsl.labels" -}}
helm.sh/chart: {{ include "neural-dsl.chart" . }}
{{ include "neural-dsl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "neural-dsl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "neural-dsl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "neural-dsl.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "neural-dsl.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image name helper
*/}}
{{- define "neural-dsl.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if .Values.global.imageRegistry }}
{{- $registry = .Values.global.imageRegistry -}}
{{- end }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}

{{/*
Redis URL helper
*/}}
{{- define "neural-dsl.redisUrl" -}}
{{- printf "redis://:%s@redis:%d/0" .Values.redis.auth.password (.Values.redis.service.port | int) }}
{{- end }}

{{/*
Database URL helper
*/}}
{{- define "neural-dsl.databaseUrl" -}}
{{- printf "postgresql://%s:%s@postgres:%d/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (.Values.postgresql.service.port | int) .Values.postgresql.auth.database }}
{{- end }}

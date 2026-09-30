from django.db import models


class AnaliseML(models.Model):
    """
    Histórico completo de cada classificação realizada pela aplicação.
    Mantém entradas, resultados do ML, resposta do Qwen e metadados para
    consultas posteriores via MCP.
    """

    criado_em = models.DateTimeField(auto_now_add=True, db_index=True)

    # Dados de entrada da análise
    comprimento_sepala = models.FloatField()
    largura_sepala = models.FloatField()
    comprimento_petala = models.FloatField()
    largura_petala = models.FloatField()

    # Resultado do modelo tradicional
    resultado_decision_tree = models.CharField(max_length=100, blank=True)
    classe_decision_tree = models.IntegerField(null=True, blank=True)
    probabilidades_decision_tree = models.JSONField(default=dict, blank=True)

    # Resultado da IA generativa
    resultado_qwen = models.TextField(blank=True)
    qwen_modelo = models.CharField(max_length=100, blank=True)
    qwen_prompt = models.TextField(blank=True)
    qwen_status = models.CharField(max_length=30, default="ok")
    qwen_tempo_ms = models.FloatField(null=True, blank=True)

    # Contexto adicional preservado para futuras consultas
    metadados = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-criado_em"]

    def __str__(self):
        return f"Análise #{self.pk} - {self.criado_em:%Y-%m-%d %H:%M:%S}"
- the imputed vcf file must have matching sample names with the truth vcf
- must bgzip and tabix the truth and imputed vcf

```bash
python -m src.python.vcf_eval.accuracy 
  --truth ../phg-cassava/cassava_pangenome_diploid.vcf.gz 
  --partial-credit 
  --missing-as-ref 
  -r Chromosome03 
  --samples 2020g_08_01,Aipim_Abacate,BGM_2104,BGM_2105,BR_11_34_41,BRS_Formosa,BRS_Jari,BRS_Novo_Horizonte,BRS_Poti_Branca,Capixaba,IITA_TMS_IBA020516,VEN25 
  --imputed ../phg-cassava/HMM-dip-0.01x.vcf.gz 
    > ../phg-cassava/HMM-dip-0.01x-acc.txt
```
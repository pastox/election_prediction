import numpy as np
import pandas as pd
import collections

## Import data

#Election Data
rawElectionData = pd.read_excel('Presidentielle_2017_Resultats_Tour_1_c_canton.xlsx', header=0, skiprows= [0,1,2])

# Export information about the cantons : equivalence between departement and cantons codes and the canton's name
cantonsData = rawElectionData[['Code du département', 'Libellé du département', 'Code du canton', 'Libellé du canton']]
cantonsData.to_csv('cantonsData.csv', sep=',', index= False)

#We will only be working with the five main candidates. We will try to predict their election results in the french "cantons"
candidates = ["MACRON", "LE PEN", "FILLON", "MÉLENCHON", "HAMON"]

# Let's split the data separately for each candidate
candidatesDatas = []
for candidate in candidates:
    candidateDatas = []
    columns = ['Code du département', 'Code du canton', 'Inscrits', 'Nom', 'Sexe', '% Voix/Exp']
    candidateDatas.append(rawElectionData[columns][rawElectionData['Nom'] == candidate])
    for i in range(1,11):
        columns = ['Code du département', 'Code du canton', 'Inscrits', 'Nom.'+str(i), 'Sexe.'+str(i), '% Voix/Exp.'+str(i)]
        candidateDatas.append(rawElectionData[columns][rawElectionData['Nom.'+str(i)] == candidate])
    candidatesDatas.append(candidateDatas)

print(candidatesDatas)

# we have to rename some columns before concatenating the data
for i in range(len(candidates)):
    for j in range(1, 11):
        candidatesDatas[i][j] = candidatesDatas[i][j].rename(index=str, columns={'Nom.'+str(j) : 'Nom', 'Sexe.'+str(j) : 'Sexe', '% Voix/Exp.'+str(j) : '% Voix/Exp'})

#concatenation of the data for each candidate
candidatesDatas2 = []
for i in range(len(candidates)):
    candidatesDatas2.append(pd.concat(candidatesDatas[i]))
    candidatesDatas2[-1] = candidatesDatas2[-1].sort_values(by=['Code du département', 'Code du canton'])
    candidatesDatas2[-1] = candidatesDatas2[-1].rename(columns={'Code du département' : 'Département', 'Code du canton' : 'Code Canton'})

    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZS']
    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZN']
    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZM']
    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZP']
    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZW']
    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZX']
    candidatesDatas2[-1] = candidatesDatas2[-1][candidatesDatas2[-1]['Département'] != 'ZZ']

    candidatesDatas2[-1] = candidatesDatas2[-1].replace('ZA', '971')
    candidatesDatas2[-1] = candidatesDatas2[-1].replace('ZB', '972')
    candidatesDatas2[-1] = candidatesDatas2[-1].replace('ZC', '973')
    candidatesDatas2[-1] = candidatesDatas2[-1].replace('ZD', '974')

    candidatesDatas2[-1]['Département'] = candidatesDatas2[-1]['Département'].astype(str)
    candidatesDatas2[-1]['Code Canton'] = candidatesDatas2[-1]['Code Canton'].astype(str)

    for j in range(candidatesDatas2[-1].shape[0]):
        if len(candidatesDatas2[-1]['Département'].values[j]) == 1:
            candidatesDatas2[-1]['Département'].values[j] = '0' + candidatesDatas2[-1]['Département'].values[j]

    del candidatesDatas2[-1]['Nom']
    del candidatesDatas2[-1]['Sexe']

print(candidatesDatas2)

#Here we have the Election Data for each candidate
macronElectionData = candidatesDatas2[0]
lepenElectionData = candidatesDatas2[1]
fillonElectionData = candidatesDatas2[2]
melenchonElectionData = candidatesDatas2[3]
hamonElectionData = candidatesDatas2[4]

# We will need to aggregate data from towns from same canton : let's load a database that says in which canton every town is
equivalenceData = pd.read_excel('Table_de_correspondance_circo_legislatives2017-1.xlsx', header=0)

equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZS']
equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZN']
equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZM']
equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZP']
equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZW']
equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZX']
equivalenceData = equivalenceData[equivalenceData['CODE DPT'] != 'ZZ']

equivalenceData = equivalenceData.replace('ZA', '97')
equivalenceData = equivalenceData.replace('ZB', '97')
equivalenceData = equivalenceData.replace('ZC', '97')
equivalenceData = equivalenceData.replace('ZD', '97')

#load of the first features dataset
townData1 = pd.read_excel('base_cc_comparateur.xls', header=0, skiprows=[0,1,2,3,5])
townData1['CODGEO'] = townData1['CODGEO'].astype(str)

#now we have to aggregate the data into cantons

codeCanton = np.zeros((townData1.shape[0],1))
for i in range(townData1.shape[0]):
    if i%1000 == 0:
        print(i)
    codeVille = str(townData1['CODGEO'].at[i])
    dep = codeVille[:2]
    if dep[0] == '0':
        dep = dep[1]
    reste = codeVille[2:]
    if reste[0] == '0':
        reste = reste[1:]
    if reste[0] == '0':
        reste = reste[1:]
    codeCantonHyp = equivalenceData['CODE CANTON'][(equivalenceData['CODE DPT'].astype(str) == dep) & (equivalenceData['CODE COMMUNE'].astype(str) == reste)]
    if codeCantonHyp.values.shape[0] > 0:
        codeCanton[i,0] = codeCantonHyp.values[0]

townData1['Code Canton'] = pd.DataFrame(codeCanton)

del townData1['Libellé commune ou ARM']
del townData1['CODGEO']
del townData1['Région']

dico = {}
for column in townData1.columns:
    dico[column] = 'sum'
dico['Médiane du niveau vie en 2015'] = 'mean'
dico['Département'] = 'max'
dico['Code Canton'] = 'max'
townData11 = townData1.groupby(['Département', 'Code Canton']).aggregate(dico)

townData11 = townData11[townData11['Superficie'] > 0]

#let's join the candidates election data frames with townData11 on the departement and canton codes

townData11['Code Canton'] = townData11['Code Canton'].astype(int)
townData11['Code Canton'] = townData11['Code Canton'].astype(str)

macronElectionCantonData1 = macronElectionData.join(other=townData11.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
lepenElectionCantonData1 = lepenElectionData.join(other=townData11.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
fillonElectionCantonData1 = fillonElectionData.join(other=townData11.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
melenchonElectionCantonData1 = melenchonElectionData.join(other=townData11.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
hamonElectionCantonData1 = hamonElectionData.join(other=townData11.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])

#Now let's load the second dataset with information about towns and aggregate according to cantons all the same
townData2 = pd.read_excel('MDB-INSEE-V2.xls', header=0)
townData2['CODGEO'] = townData2['CODGEO'].astype(str)
townData2['Code Canton'] = pd.DataFrame(codeCanton).astype(int).astype(str)
townData2['Département'] = townData2['DEP'].rename(columns={'DEP' : 'Département'})

del townData2['DEP']
del townData2['CP']
del townData2['CODGEO']
del townData2['SEG Croissance POP']
del townData2['LIBGEO']
del townData2['REG']
del townData2['Urbanité Ruralité']
del townData2['SEG Environnement Démographique Obsolète']
del townData2['Environnement Démographique']
del townData2['Fidélité']
del townData2['SYN MEDICAL']
del townData2['Seg Cap Fiscale']
del townData2['Seg Dyn Entre']
del townData2['DYN SetC']
del townData2['Orientation Economique']
del townData2['Moyenne Revenus Fiscaux Régionaux']
del townData2['Reg Moyenne Salaires Horaires']
del townData2['Reg Moyenne Salaires Cadre Horaires']
del townData2['Reg Moyenne Salaires Prof Intermédiaire Horaires']
del townData2['Reg Moyenne Salaires Employé Horaires']
del townData2['Reg Moyenne Salaires Ouvrié Horaires']
del townData2['Dynamique Démographique INSEE']
del townData2['Capacité Fisc']
del townData2['Score VA Région']
del townData2['Dynamique Démographique BV']

dico2 = {}
for column in townData2.columns:
    dico2[column] = 'sum'
dico2['Indice Fiscal Partiel'] = 'mean'
dico2['Score Fiscal'] = 'mean'
dico2['Indice Evasion Client'] = 'mean'
dico2['Score Evasion Client'] = 'mean'
dico2['Indice Synergie Médicale'] = 'mean'
dico2['Score Synergie Médicale'] = 'mean'
dico2['Densité Médicale BV'] = 'mean'
dico2['Score équipement de santé BV'] = 'mean'
dico2['Indice Démographique'] = 'mean'
dico2['Score Démographique'] = 'mean'
dico2['Indice Ménages'] = 'mean'
dico2['Score Ménages'] = 'mean'
dico2['Evolution Pop %'] = 'mean'
dico2['Moyenne Revenus Fiscaux Départementaux'] = 'max'
dico2['Valeur ajoutée régionale'] = 'max'
dico2['Score Urbanité'] = 'mean'
dico2['Taux étudiants'] = 'mean'
dico2['Taux Propriété'] = 'mean'
dico2['Moyenne Revnus fiscaux'] = 'mean'
dico2['Taux Evasion Client'] = 'mean'
dico2['Nb Industries des biens intermédiaires'] = 'max'
dico2['Nb de Commerce'] = 'max'
dico2['Nb de Services aux particuliers'] = 'max'
dico2['Nb institution de Education, santé, action sociale, administration'] = 'max'
dico2['PIB Régionnal'] = 'max'
dico2['Score Croissance Population'] = 'mean'
dico2['Score Croissance Entrepreneuriale'] = 'mean'
dico2['Score PIB'] = 'max'

dico2['Département'] = 'max'
dico2['Code Canton'] = 'max'
townData22 = townData2.groupby(['Département', 'Code Canton']).aggregate(dico2)

# Now let's merge all the datasets!
macronElectionCantonData2 = macronElectionCantonData1.join(other=townData22.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
lepenElectionCantonData2 = lepenElectionCantonData1.join(other=townData22.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
fillonElectionCantonData2 = fillonElectionCantonData1.join(other=townData22.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
melenchonElectionCantonData2 = melenchonElectionCantonData1.join(other=townData22.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])
hamonElectionCantonData2 = hamonElectionCantonData1.join(other=townData22.set_index(['Département', 'Code Canton']), how='left', on=['Département', 'Code Canton'])

#NaN Values
macronElectionCantonData3 = macronElectionCantonData2.fillna(macronElectionCantonData2.mean())
lepenElectionCantonData3 = lepenElectionCantonData2.fillna(macronElectionCantonData2.mean())
fillonElectionCantonData3 = fillonElectionCantonData2.fillna(macronElectionCantonData2.mean())
melenchonElectionCantonData3 = melenchonElectionCantonData2.fillna(macronElectionCantonData2.mean())
hamonElectionCantonData3 = hamonElectionCantonData2.fillna(macronElectionCantonData2.mean())

#Normalization between 0 and 1 except for Département, Code Canton and % Voix/Exp, which we just divide by 100
macronDepCanton = macronElectionCantonData3[['Département', 'Code Canton', '% Voix/Exp']]
del macronElectionCantonData3['Département']
del macronElectionCantonData3['Code Canton']
del macronElectionCantonData3['% Voix/Exp']
macronElectionCantonData4 = ( macronElectionCantonData3.astype(float) - macronElectionCantonData3.astype(float).min() ) / ( macronElectionCantonData3.astype(float).max() - macronElectionCantonData3.astype(float).min() )
macronElectionCantonData4[['Département', 'Code Canton', '% Voix/Exp']] = macronDepCanton
macronElectionCantonData4['% Voix/Exp'] = macronElectionCantonData4['% Voix/Exp']/100.0

lepenDepCanton = lepenElectionCantonData3[['Département', 'Code Canton', '% Voix/Exp']]
del lepenElectionCantonData3['Département']
del lepenElectionCantonData3['Code Canton']
del lepenElectionCantonData3['% Voix/Exp']
lepenElectionCantonData4 = ( lepenElectionCantonData3.astype(float) - lepenElectionCantonData3.astype(float).min() ) / ( lepenElectionCantonData3.astype(float).max() - lepenElectionCantonData3.astype(float).min() )
lepenElectionCantonData4[['Département', 'Code Canton', '% Voix/Exp']] = lepenDepCanton
lepenElectionCantonData4['% Voix/Exp'] = lepenElectionCantonData4['% Voix/Exp']/100.0

fillonDepCanton = fillonElectionCantonData3[['Département', 'Code Canton', '% Voix/Exp']]
del fillonElectionCantonData3['Département']
del fillonElectionCantonData3['Code Canton']
del fillonElectionCantonData3['% Voix/Exp']
fillonElectionCantonData4 = ( fillonElectionCantonData3.astype(float) - fillonElectionCantonData3.astype(float).min() ) / ( fillonElectionCantonData3.astype(float).max() - fillonElectionCantonData3.astype(float).min() )
fillonElectionCantonData4[['Département', 'Code Canton', '% Voix/Exp']] = fillonDepCanton
fillonElectionCantonData4['% Voix/Exp'] = fillonElectionCantonData4['% Voix/Exp']/100.0

melenchonDepCanton = melenchonElectionCantonData3[['Département', 'Code Canton', '% Voix/Exp']]
del melenchonElectionCantonData3['Département']
del melenchonElectionCantonData3['Code Canton']
del melenchonElectionCantonData3['% Voix/Exp']
melenchonElectionCantonData4 = ( melenchonElectionCantonData3.astype(float) - melenchonElectionCantonData3.astype(float).min() ) / ( melenchonElectionCantonData3.astype(float).max() - melenchonElectionCantonData3.astype(float).min() )
melenchonElectionCantonData4[['Département', 'Code Canton', '% Voix/Exp']] = melenchonDepCanton
melenchonElectionCantonData4['% Voix/Exp'] = melenchonElectionCantonData4['% Voix/Exp']/100.0

hamonDepCanton = hamonElectionCantonData3[['Département', 'Code Canton', '% Voix/Exp']]
del hamonElectionCantonData3['Département']
del hamonElectionCantonData3['Code Canton']
del hamonElectionCantonData3['% Voix/Exp']
hamonElectionCantonData4 = ( hamonElectionCantonData3.astype(float) - hamonElectionCantonData3.astype(float).min() ) / ( hamonElectionCantonData3.astype(float).max() - hamonElectionCantonData3.astype(float).min() )
hamonElectionCantonData4[['Département', 'Code Canton', '% Voix/Exp']] = hamonDepCanton
hamonElectionCantonData4['% Voix/Exp'] = hamonElectionCantonData4['% Voix/Exp']/100.0

#Feature Selection : a filter approach : f_regression (linear method): feature selection on macron's dataset and then we take the same features for all the candidates

def feature_selection(k):

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression

    yMacron = macronElectionCantonData4['% Voix/Exp'].astype(float)
    XMacron = macronElectionCantonData4.copy()
    del XMacron['% Voix/Exp']
    del XMacron['Département']
    del XMacron['Code Canton']
    XMacron = XMacron.astype(float)

    yLepen = lepenElectionCantonData4['% Voix/Exp'].astype(float)
    XLepen = lepenElectionCantonData4.copy()
    del XLepen['% Voix/Exp']
    del XLepen['Département']
    del XLepen['Code Canton']
    XLepen = XLepen.astype(float)

    yFillon = fillonElectionCantonData4['% Voix/Exp'].astype(float)
    XFillon = fillonElectionCantonData4.copy()
    del XFillon['% Voix/Exp']
    del XFillon['Département']
    del XFillon['Code Canton']
    XFillon = XFillon.astype(float)

    yMelenchon = melenchonElectionCantonData4['% Voix/Exp'].astype(float)
    XMelenchon = melenchonElectionCantonData4.copy()
    del XMelenchon['% Voix/Exp']
    del XMelenchon['Département']
    del XMelenchon['Code Canton']
    XMelenchon = XMelenchon.astype(float)

    yHamon = hamonElectionCantonData4['% Voix/Exp'].astype(float)
    XHamon = hamonElectionCantonData4.copy()
    del XHamon['% Voix/Exp']
    del XHamon['Département']
    del XHamon['Code Canton']
    XHamon = XHamon.astype(float)


    macronSelectedFeatures = list(XMacron.columns[SelectKBest(f_regression, k=k).fit(XMacron, yMacron).get_support(indices=True)])
    lepenSelectedFeatures = list(XLepen.columns[SelectKBest(f_regression, k=k).fit(XLepen, yLepen).get_support(indices=True)])
    fillonSelectedFeatures = list(XFillon.columns[SelectKBest(f_regression, k=k).fit(XFillon, yFillon).get_support(indices=True)])
    melenchonSelectedFeatures = list(XMelenchon.columns[SelectKBest(f_regression, k=k).fit(XMelenchon, yMelenchon).get_support(indices=True)])
    hamonSelectedFeatures = list(XHamon.columns[SelectKBest(f_regression, k=k).fit(XHamon, yHamon).get_support(indices=True)])

    selectedFeatures = []
    for feature in macronSelectedFeatures+lepenSelectedFeatures+fillonSelectedFeatures+melenchonSelectedFeatures+hamonSelectedFeatures:
        if not(feature in selectedFeatures):
            selectedFeatures.append(feature)

    XMacron_new = XMacron[selectedFeatures]
    XLepen_new = XLepen[selectedFeatures]
    XFillon_new = XFillon[selectedFeatures]
    XMelenchon_new = XMelenchon[selectedFeatures]
    XHamon_new = XHamon[selectedFeatures]

    XMacron_new[['Département', 'Code Canton', '% Voix/Exp']] = macronDepCanton
    XMacron_new['% Voix/Exp'] = XMacron_new['% Voix/Exp']/100.0
    XLepen_new[['Département', 'Code Canton', '% Voix/Exp']] = lepenDepCanton
    XLepen_new['% Voix/Exp'] = XLepen_new['% Voix/Exp']/100.0
    XFillon_new[['Département', 'Code Canton', '% Voix/Exp']] = fillonDepCanton
    XFillon_new['% Voix/Exp'] = XFillon_new['% Voix/Exp']/100.0
    XMelenchon_new[['Département', 'Code Canton', '% Voix/Exp']] = melenchonDepCanton
    XMelenchon_new['% Voix/Exp'] = XMelenchon_new['% Voix/Exp']/100.0
    XHamon_new[['Département', 'Code Canton', '% Voix/Exp']] = hamonDepCanton
    XHamon_new['% Voix/Exp'] = XHamon_new['% Voix/Exp']/100.0

    return XMacron_new, XLepen_new, XFillon_new, XMelenchon_new, XHamon_new

macronData, lepenData, fillonData, melenchonData, hamonData = feature_selection(20)

##Export to csv
macronData.to_csv('macronData.csv', sep=',', index= False)
lepenData.to_csv('lepenData.csv', sep=',', index= False)
fillonData.to_csv('fillonData.csv', sep=',', index= False)
melenchonData.to_csv('melenchonData.csv', sep=',', index=False)
hamonData.to_csv('hamonData.csv', sep=',', index= False)

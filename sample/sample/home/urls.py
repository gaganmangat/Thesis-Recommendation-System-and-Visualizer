from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
path('autosuggest', views.autosuggest, name = 'autosuggest'),

path('getidrec', views.getidrec, name = 'getidrec'),

path('cleanrepo', views.cleanrepo, name = 'cleanrepo'),
path('cserepo', views.cserepo, name = 'cserepo'),

path('',views.home,name='home'),
path('analysis',views.analysis,name='analysis'),
path('analytics',views.analytics,name='analytics'),
path('contact',views.contact,name='contact'),
path('recom',views.recom,name='recom'),
path('repo',views.repo,name='repo'),
path('splash',views.splash,name='splash'),
path('analyhome',views.analyhome,name='analyhome'),
path('cseanalytics',views.cseanalytics,name='cseanalytics'),

path('link',views.link,name='link'),

path('ae',views.ae,name='ae'),
path('bsbe',views.bsbe,name='bsbe'),
path('ce',views.ce,name='ce'),
path('celp',views.celp,name='celp'),
path('che',views.che,name='che'),
path('chm',views.chm,name='chm'),
path('civil',views.civil,name='civil'),
path('des',views.des,name='des'),
path('dp',views.dp,name='dp'),
path('eco',views.eco,name='eco'),
path('ee',views.ee,name='ee'),
path('eem',views.eem,name='eem'),
path('eemp',views.eemp,name='eemp'),
path('es',views.es,name='es'),
path('hss',views.hss,name='hss'),
path('ime',views.ime,name='ime'),
path('lt',views.lt,name='lt'),
path('ltp',views.ltp,name='ltp'),
path('math',views.math,name='math'),
path('me',views.me,name='me'),
path('mme',views.mme,name='mme'),
path('mse',views.mse,name='mse'),
path('msp',views.msp,name='msp'),
path('net',views.net,name='net'),
path('netp',views.netp,name='netp'),
path('phy',views.phy,name='phy'),

path('recommend_results',views.recommend_results,name='recommend_results'),


path('aerepo',views.aerepo,name='aerepo'),
path('bsberepo',views.bsberepo,name='bsberepo'),
path('cerepo',views.cerepo,name='cerepo'),
path('celprepo',views.celprepo,name='celprepo'),
path('cherepo',views.cherepo,name='cherepo'),
path('chmrepo',views.chmrepo,name='chmrepo'),
path('civilrepo',views.civilrepo,name='civilrepo'),
path('desrepo',views.desrepo,name='desrepo'),
path('dprepo',views.dprepo,name='dprepo'),
path('ecorepo',views.ecorepo,name='ecorepo'),
path('eerepo',views.eerepo,name='eerepo'),
path('eemrepo',views.eemrepo,name='eemrepo'),
path('esrepo',views.esrepo,name='esrepo'),
path('hssrepo',views.hssrepo,name='hssrepo'),
path('imerepo',views.imerepo,name='imerepo'),
path('ltrepo',views.ltrepo,name='ltrepo'),
path('ltprepo',views.ltprepo,name='ltprepo'),
path('mathrepo',views.mathrepo,name='mathrepo'),
path('merepo',views.merepo,name='merepo'),
path('mmerepo',views.mmerepo,name='mmerepo'),
path('msprepo',views.msprepo,name='msprepo'),
path('netrepo',views.netrepo,name='netrepo'),
path('netprepo',views.netprepo,name='netprepo'),
path('phyrepo',views.phyrepo,name='phyrepo'),


#path('splash',views.home,name='splash'),
]

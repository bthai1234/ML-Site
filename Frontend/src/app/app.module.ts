import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NbThemeModule, NbLayoutModule, NbSidebarModule, NbButtonModule, NbMenuModule, NbSidebarService, NbMenuService } from '@nebular/theme';
import { NbEvaIconsModule } from '@nebular/eva-icons';
import { ModelsPageComponent } from './models-page/models-page.component';

@NgModule({
  declarations: [
    AppComponent,
    ModelsPageComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    NbThemeModule.forRoot({ name: 'dark' }),
    NbLayoutModule,
    NbEvaIconsModule,
    NbMenuModule.forRoot(),
    NbSidebarModule,
    NbButtonModule,

  ],
  providers: [NbSidebarService, NbMenuService],
  bootstrap: [AppComponent]
})
export class AppModule { }

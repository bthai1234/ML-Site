import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ModelsPageComponent } from './models-page/models-page.component';

const routes: Routes = [
  { path: '', redirectTo: '/models', pathMatch: 'full' },
  {path: 'models', component: ModelsPageComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes, { useHash: true })],
  exports: [RouterModule]
})
export class AppRoutingModule { }

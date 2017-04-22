import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { HttpModule } from '@angular/http';

import { AppComponent } from './app.component.js';
import { LoginComponent } from './login/login.component.js';
import { NavbarComponent } from './navbar/navbar.component.js'
import { HomeComponent } from './home/home.component.js'
import { CreateAccountComponent } from './create-account/create-account.component.js'
import { SearchResultComponent } from './search-result/search-result.component.js'

@NgModule({
  imports: [
    BrowserModule,
    FormsModule,
    HttpModule,
    RouterModule.forRoot([
      { path: 'login', component: LoginComponent },
      { path: 'home', component: HomeComponent },
      { path: 'createaccount', component: CreateAccountComponent },
      { path: 'searchresult', component: SearchResultComponent },
      { path: '', redirectTo: 'home', pathMatch: 'full' },
      { path: '**', redirectTo: 'home', pathMatch: 'full' }
    ], { useHash: true })
  ],
  declarations: [AppComponent, LoginComponent, NavbarComponent, HomeComponent, CreateAccountComponent, SearchResultComponent],
  bootstrap: [AppComponent]
})
export class AppModule { }
